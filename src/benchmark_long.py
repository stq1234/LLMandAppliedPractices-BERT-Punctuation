"""Compare the original serial punctuation window with batched fixed windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, BertForTokenClassification

from train import LABELS, MODEL_NAME, MODEL_REVISION, ROOT, confusion_metrics, read_jsonl


DEFAULT_BATCH_SIZES = (1, 4, 8, 16)
SERIAL_MAX_TOKENS = 400
SERIAL_CONTENT_LIMIT = SERIAL_MAX_TOKENS - 2
SERIAL_COMMIT_LIMIT = 300
LONG_TEXT_MIN_CHARS = 800


@dataclass
class BenchWindow:
    article_id: str
    start: int
    end: int
    input_ids: list[int]


def make_starts(length: int, window_size: int, stride: int) -> list[int]:
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    last_start = length - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    if starts[0] != 0 or any(right <= left for left, right in zip(starts, starts[1:])):
        raise AssertionError("Fixed-window starts are not strictly ordered from zero")
    if any(right > left + window_size for left, right in zip(starts, starts[1:])):
        raise AssertionError("Fixed-window plan has uncovered character gaps")
    return starts


def make_plan(
    articles: list[dict[str, Any]],
    tokenizer: Any,
    window_size: int,
    stride: int,
) -> tuple[list[BenchWindow], dict[str, list[int]]]:
    windows: list[BenchWindow] = []
    ids_by_article: dict[str, list[int]] = {}
    cls_id = int(tokenizer.cls_token_id)
    sep_id = int(tokenizer.sep_token_id)
    for article in articles:
        article_id = str(article["id"])
        text = str(article["text"])
        char_ids = tokenizer.convert_tokens_to_ids(list(text))
        if len(char_ids) != len(text):
            raise AssertionError(f"Tokenizer did not preserve per-character alignment for {article_id}")
        ids_by_article[article_id] = char_ids
        for start in make_starts(len(char_ids), window_size, stride):
            end = min(start + window_size, len(char_ids))
            windows.append(
                BenchWindow(article_id, start, end, [cls_id, *char_ids[start:end], sep_id])
            )
        coverage = np.zeros(len(text), dtype=np.uint8)
        for window in (item for item in windows if item.article_id == article_id):
            coverage[window.start : window.end] += 1
        if np.any(coverage == 0):
            raise AssertionError(f"Fixed windows left holes in {article_id}")
    return windows, ids_by_article


def collate_plan(windows: list[BenchWindow], pad_id: int) -> dict[str, torch.Tensor]:
    max_length = max(len(window.input_ids) for window in windows)
    input_ids = []
    attention_mask = []
    for window in windows:
        padding = max_length - len(window.input_ids)
        input_ids.append(window.input_ids + [pad_id] * padding)
        attention_mask.append([1] * len(window.input_ids) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def plan_batches(
    windows: list[BenchWindow], pad_id: int, batch_size: int, device: torch.device
) -> list[tuple[list[BenchWindow], torch.Tensor, torch.Tensor]]:
    batches = []
    for start in range(0, len(windows), batch_size):
        group = windows[start : start + batch_size]
        cpu_batch = collate_plan(group, pad_id)
        batches.append(
            (
                group,
                cpu_batch["input_ids"].to(device, non_blocking=True),
                cpu_batch["attention_mask"].to(device, non_blocking=True),
            )
        )
    torch.cuda.synchronize(device)
    return batches


def fuse_probabilities(
    articles: list[dict[str, Any]],
    windows: list[BenchWindow],
    probability_rows: list[np.ndarray],
) -> dict[str, list[int]]:
    sums = {
        str(article["id"]): np.zeros((len(str(article["text"])), len(LABELS)), dtype=np.float64)
        for article in articles
    }
    counts = {
        str(article["id"]): np.zeros(len(str(article["text"])), dtype=np.uint16)
        for article in articles
    }
    for window, probabilities in zip(windows, probability_rows):
        width = window.end - window.start
        sums[window.article_id][window.start : window.end] += probabilities[:width]
        counts[window.article_id][window.start : window.end] += 1
    predictions: dict[str, list[int]] = {}
    for article in articles:
        article_id = str(article["id"])
        if np.any(counts[article_id] == 0):
            raise AssertionError(f"Probability fusion left holes in {article_id}")
        predictions[article_id] = np.argmax(
            sums[article_id] / counts[article_id][:, None], axis=-1
        ).astype(np.int16).tolist()
    return predictions


def metrics_for(
    articles: list[dict[str, Any]],
    predictions: dict[str, list[int]],
) -> tuple[dict[str, Any], np.ndarray]:
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for article in articles:
        article_id = str(article["id"])
        gold = np.asarray(article["labels"], dtype=np.int64)
        predicted = np.asarray(predictions[article_id], dtype=np.int64)
        if len(gold) != len(predicted):
            raise AssertionError(f"Prediction length mismatch for {article_id}")
        confusion += np.bincount(
            gold * len(LABELS) + predicted,
            minlength=len(LABELS) * len(LABELS),
        ).reshape(len(LABELS), len(LABELS))
    return confusion_metrics(confusion), confusion


def fixed_predict_from_plan(
    model: torch.nn.Module,
    articles: list[dict[str, Any]],
    windows: list[BenchWindow],
    batches: list[tuple[list[BenchWindow], torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    model.eval()
    probability_rows: list[np.ndarray] = []
    forward_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    processed_tokens_nonpad = sum(len(window.input_ids) for window in windows)
    processed_tokens_padded = sum(ids.numel() for _, ids, _ in batches)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    wall_started = time.perf_counter()
    with torch.inference_mode():
        for _, input_ids, attention_mask in batches:
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            event_start.record()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            event_end.record()
            forward_events.append((event_start, event_end))
            # Position 0 is [CLS]; restore probabilities in the same order as
            # global character offsets by dropping [CLS] before fusion.
            probability_rows.extend(
                torch.softmax(logits.float()[:, 1:, :], dim=-1).cpu().numpy()
            )
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_started
    gpu_seconds = sum(event_start.elapsed_time(event_end) / 1000.0 for event_start, event_end in forward_events)
    predictions = fuse_probabilities(articles, windows, probability_rows)
    return predictions, {
        "forward_wall_seconds": wall_seconds,
        "forward_gpu_seconds": gpu_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "window_count": len(windows),
        "input_tokens_nonpad": processed_tokens_nonpad,
        "input_tokens_padded": processed_tokens_padded,
    }


def fixed_end_to_end(
    model: torch.nn.Module,
    articles: list[dict[str, Any]],
    tokenizer: Any,
    window_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, list[int]], dict[str, Any], list[BenchWindow]]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    windows, _ = make_plan(articles, tokenizer, window_size, stride)
    prediction_rows: list[np.ndarray] = []
    processed_tokens_nonpad = sum(len(window.input_ids) for window in windows)
    processed_tokens_padded = 0
    model_gpu_seconds = 0.0
    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, len(windows), batch_size):
            group = windows[batch_start : batch_start + batch_size]
            cpu_batch = collate_plan(group, int(tokenizer.pad_token_id))
            processed_tokens_padded += int(cpu_batch["input_ids"].numel())
            input_ids = cpu_batch["input_ids"].to(device, non_blocking=True)
            attention_mask = cpu_batch["attention_mask"].to(device, non_blocking=True)
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            event_start.record()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            event_end.record()
            torch.cuda.synchronize(device)
            model_gpu_seconds += event_start.elapsed_time(event_end) / 1000.0
            # Position 0 is [CLS]; predictions are fused against character
            # offsets, so skip it (the final [SEP] is ignored by window width).
            prediction_rows.extend(
                torch.softmax(logits.float()[:, 1:, :], dim=-1).cpu().numpy()
            )
    predictions = fuse_probabilities(articles, windows, prediction_rows)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return predictions, {
        "end_to_end_seconds": elapsed,
        "model_gpu_seconds": model_gpu_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "window_count": len(windows),
        "input_tokens_nonpad": processed_tokens_nonpad,
        "input_tokens_padded": processed_tokens_padded,
    }, windows


def original_serial_end_to_end(
    model: torch.nn.Module,
    articles: list[dict[str, Any]],
    tokenizer: Any,
    device: torch.device,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    predictions: dict[str, list[int]] = {}
    window_count = 0
    input_tokens_nonpad = 0
    forward_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    cls_id = int(tokenizer.cls_token_id)
    sep_id = int(tokenizer.sep_token_id)
    pad_id = int(tokenizer.pad_token_id)
    model.eval()
    with torch.inference_mode():
        for article in articles:
            article_id = str(article["id"])
            text = str(article["text"])
            char_ids = tokenizer.convert_tokens_to_ids(list(text))
            predicted_article = [-1] * len(text)
            cursor = 0
            while cursor < len(text):
                end = min(cursor + SERIAL_CONTENT_LIMIT, len(text))
                content_ids = char_ids[cursor:end]
                sequence = [cls_id, *content_ids, sep_id]
                padding = SERIAL_MAX_TOKENS - len(sequence)
                input_ids = torch.tensor(
                    [sequence + [pad_id] * padding], dtype=torch.long, device=device
                )
                attention_mask = torch.tensor(
                    [[1] * len(sequence) + [0] * padding], dtype=torch.long, device=device
                )
                event_start = torch.cuda.Event(enable_timing=True)
                event_end = torch.cuda.Event(enable_timing=True)
                event_start.record()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                event_end.record()
                forward_events.append((event_start, event_end))
                current_predictions = logits[0, 1 : 1 + len(content_ids)].argmax(dim=-1).cpu().tolist()
                remaining = len(text) - cursor
                if remaining <= SERIAL_CONTENT_LIMIT:
                    commit_count = remaining
                else:
                    inspect_count = min(SERIAL_COMMIT_LIMIT, len(current_predictions))
                    period_positions = [
                        index
                        for index, label_id in enumerate(current_predictions[:inspect_count])
                        if label_id == 2
                    ]
                    commit_count = period_positions[-1] + 1 if period_positions else SERIAL_COMMIT_LIMIT
                predicted_article[cursor : cursor + commit_count] = current_predictions[:commit_count]
                cursor += commit_count
                window_count += 1
                input_tokens_nonpad += len(sequence)
            if any(label_id < 0 for label_id in predicted_article):
                raise AssertionError(f"Original serial windows left uncovered characters in {article_id}")
            predictions[article_id] = predicted_article
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    forward_seconds = sum(event_start.elapsed_time(event_end) / 1000.0 for event_start, event_end in forward_events)
    return predictions, {
        "end_to_end_seconds": elapsed,
        "model_gpu_seconds": forward_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "window_count": window_count,
        "input_tokens_nonpad": input_tokens_nonpad,
        "input_tokens_padded": window_count * SERIAL_MAX_TOKENS,
    }


def window_plan_sha256(windows: list[BenchWindow]) -> str:
    canonical = "\n".join(
        f"{window.article_id}\t{window.start}\t{window.end}" for window in windows
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def summarize_repetitions(
    method: str,
    records: list[dict[str, Any]],
    unique_chars: int,
) -> dict[str, Any]:
    is_forward_only = method.startswith("fixed_forward_")
    e2e_key = "forward_gpu_seconds" if is_forward_only else "end_to_end_seconds"
    times = [float(row[e2e_key]) for row in records]
    result: dict[str, Any] = {
        "method": method,
        "repetitions": len(records),
        "timing_mode": "pretokenized, presplit, pretransferred model GPU forward only" if is_forward_only else "end-to-end",
        "median_seconds": statistics.median(times),
        "chars_per_second": unique_chars / statistics.median(times),
        "median_model_gpu_seconds": statistics.median(
            float(row.get("model_gpu_seconds", row.get("forward_gpu_seconds", 0.0)))
            for row in records
        ),
        "peak_allocated_mib": max(float(row["peak_allocated_mib"]) for row in records),
        "window_count": int(statistics.median(int(row["window_count"]) for row in records)),
        "input_tokens_nonpad": int(statistics.median(int(row["input_tokens_nonpad"]) for row in records)),
        "input_tokens_padded": int(statistics.median(int(row["input_tokens_padded"]) for row in records)),
        "raw_repetition_seconds": times,
    }
    if is_forward_only:
        result["median_python_loop_wall_seconds"] = statistics.median(
            float(row["forward_wall_seconds"]) for row in records
        )
    result["padded_tokens_per_second"] = result["input_tokens_padded"] / result["median_seconds"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCH_SIZES))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the requested GPU benchmark")
    device = torch.device("cuda")
    checkpoint = ROOT / "checkpoints" / "best_model"
    if not checkpoint.exists():
        raise FileNotFoundError("Train and select the best validation checkpoint before benchmarking")
    model = BertForTokenClassification.from_pretrained(checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    val_articles_all = read_jsonl(ROOT / "data" / "validation.jsonl")
    test_articles_all = read_jsonl(ROOT / "data" / "test.jsonl")
    val_long = [article for article in val_articles_all if len(article["text"]) >= LONG_TEXT_MIN_CHARS]
    test_long = [article for article in test_articles_all if len(article["text"]) >= LONG_TEXT_MIN_CHARS]
    if not val_long or not test_long:
        raise RuntimeError(
            f"Need long articles in validation and test; got val={len(val_long)} test={len(test_long)}"
        )

    grid = ((256, 128), (400, 300), (256, 256), (400, 400))
    grid_rows = []
    val_predictions: dict[tuple[int, int], dict[str, list[int]]] = {}
    for window_size, stride in grid:
        predictions, timing, windows = fixed_end_to_end(
            model,
            val_long,
            tokenizer,
            window_size,
            stride,
            batch_size=min(8, max(args.batch_sizes)),
            device=device,
        )
        metrics, _ = metrics_for(val_long, predictions)
        grid_rows.append(
            {
                "window_size_chars": window_size,
                "stride_chars": stride,
                "validation_long_articles": len(val_long),
                "validation_long_chars": sum(len(article["text"]) for article in val_long),
                "window_count": len(windows),
                "punctuation_micro_f1": metrics["punctuation_micro_f1"],
                "punctuation_macro_f1": metrics["punctuation_macro_f1"],
                "sentence_boundary_f1": metrics["sentence_boundary_f1"],
                "metrics": metrics,
                "validation_end_to_end_seconds_single_run": timing["end_to_end_seconds"],
            }
        )
        val_predictions[(window_size, stride)] = predictions
    selected = max(
        grid_rows,
        key=lambda row: (row["punctuation_micro_f1"], -row["window_count"]),
    )
    selected_window_size = int(selected["window_size_chars"])
    selected_stride = int(selected["stride_chars"])
    validation_result = {
        "selection_split": "validation only",
        "selection_metric": "punctuation micro-F1 excluding NONE; ties prefer fewer windows",
        "minimum_text_length_chars": LONG_TEXT_MIN_CHARS,
        "long_validation_articles": len(val_long),
        "grid": grid_rows,
        "selected": {
            "window_size_chars": selected_window_size,
            "stride_chars": selected_stride,
            "punctuation_micro_f1": selected["punctuation_micro_f1"],
            "punctuation_macro_f1": selected["punctuation_macro_f1"],
            "sentence_boundary_f1": selected["sentence_boundary_f1"],
        },
    }
    (ROOT / "results" / "long_validation_grid.json").write_text(
        json.dumps(validation_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (ROOT / "results" / "long_validation_grid.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "window_size_chars",
                "stride_chars",
                "validation_long_articles",
                "validation_long_chars",
                "window_count",
                "punctuation_micro_f1",
                "punctuation_macro_f1",
                "sentence_boundary_f1",
                "validation_end_to_end_seconds_single_run",
            ],
        )
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in grid_rows)

    selected_windows, _ = make_plan(
        test_long, tokenizer, selected_window_size, selected_stride
    )
    selected_hash = window_plan_sha256(selected_windows)
    unique_chars = sum(len(article["text"]) for article in test_long)
    staged_batches = {
        batch_size: plan_batches(
            selected_windows,
            int(tokenizer.pad_token_id),
            batch_size,
            device,
        )
        for batch_size in args.batch_sizes
    }

    # A short warmup uses the same model and input shape before any timings.
    with torch.inference_mode():
        warmup = collate_plan(selected_windows[: min(4, len(selected_windows))], int(tokenizer.pad_token_id))
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = model(
                input_ids=warmup["input_ids"].to(device),
                attention_mask=warmup["attention_mask"].to(device),
            ).logits
    torch.cuda.synchronize(device)

    forward_records: dict[int, list[dict[str, Any]]] = {batch_size: [] for batch_size in args.batch_sizes}
    e2e_records: dict[int, list[dict[str, Any]]] = {batch_size: [] for batch_size in args.batch_sizes}
    forward_predictions_by_batch: dict[int, dict[str, list[int]]] = {}
    fixed_predictions_by_batch: dict[int, dict[str, list[int]]] = {}
    serial_records: list[dict[str, Any]] = []
    serial_predictions: dict[str, list[int]] | None = None
    batch_failures: dict[int, str] = {}
    interleaved_orders = [
        list(args.batch_sizes),
        list(reversed(args.batch_sizes)),
        list(args.batch_sizes[1::2]) + list(args.batch_sizes[::2]),
    ]
    for repetition_index, order in enumerate(interleaved_orders):
        for method_index, batch_size in enumerate(order):
            if batch_size not in batch_failures:
                try:
                    forward_prediction, stats = fixed_predict_from_plan(
                        model,
                        test_long,
                        selected_windows,
                        staged_batches[batch_size],
                        device,
                    )
                    forward_records[batch_size].append(stats)
                    if repetition_index == 0:
                        forward_predictions_by_batch[batch_size] = forward_prediction
                    predictions, e2e_stats, new_windows = fixed_end_to_end(
                        model,
                        test_long,
                        tokenizer,
                        selected_window_size,
                        selected_stride,
                        batch_size,
                        device,
                    )
                    if window_plan_sha256(new_windows) != selected_hash:
                        raise AssertionError("End-to-end window plan changed across batch sizes or repetitions")
                    e2e_records[batch_size].append(e2e_stats)
                    if repetition_index == 0:
                        fixed_predictions_by_batch[batch_size] = predictions
                except torch.cuda.OutOfMemoryError as error:
                    batch_failures[batch_size] = str(error).splitlines()[0]
                    forward_records[batch_size].clear()
                    e2e_records[batch_size].clear()
                    fixed_predictions_by_batch.pop(batch_size, None)
                    forward_predictions_by_batch.pop(batch_size, None)
                    torch.cuda.empty_cache()
            # Run the sequential baseline once in the middle of every round so
            # fixed batch sizes are interleaved around its adaptive windows.
            if method_index == len(order) // 2:
                current_serial, current_stats = original_serial_end_to_end(
                    model, test_long, tokenizer, device
                )
                serial_records.append(current_stats)
                if serial_predictions is None:
                    serial_predictions = current_serial

    if not fixed_predictions_by_batch:
        raise RuntimeError("All fixed-window batch sizes ran out of GPU memory")
    fixed_reference_batch = min(fixed_predictions_by_batch)
    fixed_reference = fixed_predictions_by_batch[fixed_reference_batch]
    fixed_agreements = {}
    forward_e2e_differences = {}
    for batch_size, predictions in fixed_predictions_by_batch.items():
        changed = sum(
            a != b
            for article in test_long
            for a, b in zip(fixed_reference[str(article["id"])], predictions[str(article["id"])])
        )
        fixed_agreements[str(batch_size)] = {
            "positions_different_from_reference_batch": int(changed),
            "agreement_with_reference_batch": 1.0 - changed / unique_chars,
        }
        forward_changed = sum(
            a != b
            for article in test_long
            for a, b in zip(
                forward_predictions_by_batch[batch_size][str(article["id"])],
                predictions[str(article["id"])],
            )
        )
        forward_e2e_differences[str(batch_size)] = {
            "positions_different": int(forward_changed),
            "agreement": 1.0 - forward_changed / unique_chars,
        }
    if serial_predictions is None:
        raise AssertionError("Serial baseline failed to return predictions")
    serial_fixed_different = sum(
        a != b
        for article in test_long
        for a, b in zip(
            serial_predictions[str(article["id"])], fixed_reference[str(article["id"])]
        )
    )
    serial_metrics, _ = metrics_for(test_long, serial_predictions)
    fixed_metrics_by_batch = {
        str(batch_size): metrics_for(test_long, predictions)[0]
        for batch_size, predictions in fixed_predictions_by_batch.items()
    }
    serial_fixed_agreement = 1.0 - serial_fixed_different / unique_chars

    runtime_rows = []
    for batch_size in args.batch_sizes:
        if not forward_records[batch_size]:
            continue
        runtime_rows.append(
            summarize_repetitions(
                f"fixed_forward_bs{batch_size}",
                forward_records[batch_size],
                unique_chars,
            )
        )
        runtime_rows.append(
            summarize_repetitions(
                f"fixed_e2e_bs{batch_size}_e2e",
                e2e_records[batch_size],
                unique_chars,
            )
        )
    runtime_rows.append(
        summarize_repetitions("original_serial_e2e", serial_records, unique_chars)
    )
    runtime_rows[-1]["timing_mode"] = "adaptive serial: split/tokenize/transfer/forward/commit/fuse"
    # Keep a separately timed model-forward estimate for the serial calls.
    runtime_rows[-1]["median_model_gpu_seconds"] = statistics.median(
        float(row["model_gpu_seconds"]) for row in serial_records
    )
    runtime_data = {
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "checkpoint": "checkpoints/best_model",
        "batch_sizes": list(args.batch_sizes),
        "batch_size_failures": batch_failures,
        "warmup": "one short CUDA forward pass at the selected fixed window size; cuda synchronized before measurements",
        "repetitions": 3,
        "ordering": "batch sizes interleaved in forward/reverse/rotated order; serial baseline runs once per round between fixed methods",
        "test_long_subset_minimum_chars": LONG_TEXT_MIN_CHARS,
        "test_long_article_count": len(test_long),
        "test_long_unique_character_positions": unique_chars,
        "selected_fixed_window": {
            "window_size_chars": selected_window_size,
            "stride_chars": selected_stride,
            "merge": "arithmetic mean of class probabilities across all windows covering each global character position",
            "window_plan_sha256": selected_hash,
        },
        "original_serial_policy": {
            "input_max_tokens": SERIAL_MAX_TOKENS,
            "content_max_chars": SERIAL_CONTENT_LIMIT,
            "commit_policy": "when more than 398 chars remain, commit through the last predicted 。 among the first 300 chars, or commit 300 if none; final <=398-char remainder is retained",
            "gold_labels_used_for_window_boundaries": False,
        },
        "runtime_rows": runtime_rows,
        "test_long_subset_diagnostics": {
            "selection_used_test": False,
            "serial_metrics": serial_metrics,
            "fixed_metrics_by_batch": fixed_metrics_by_batch,
            "batch_prediction_agreement_vs_reference": fixed_agreements,
            "model_only_vs_end_to_end_prediction_agreement": forward_e2e_differences,
            "reference_batch_size": fixed_reference_batch,
            "serial_vs_fixed_batch1_positions_different": int(serial_fixed_different),
            "serial_vs_fixed_batch1_agreement": serial_fixed_agreement,
        },
    }
    (ROOT / "results" / "long_runtime_benchmark.json").write_text(
        json.dumps(runtime_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (ROOT / "results" / "long_runtime_benchmark.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        columns = [
            "method",
            "timing_mode",
            "repetitions",
            "median_seconds",
            "chars_per_second",
            "median_model_gpu_seconds",
            "median_python_loop_wall_seconds",
            "window_count",
            "input_tokens_nonpad",
            "input_tokens_padded",
            "padded_tokens_per_second",
            "peak_allocated_mib",
            "raw_repetition_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in runtime_rows:
            writer.writerow({**row, "raw_repetition_seconds": ";".join(map(str, row["raw_repetition_seconds"]))})

    with (ROOT / "results" / "long_test_predictions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for article in test_long:
            article_id = str(article["id"])
            row = {
                "article_id": article_id,
                "text": article["text"],
                "gold_label_ids": article["labels"],
                "original_serial_label_ids": serial_predictions[article_id],
                "fixed_window_batch1_label_ids": fixed_reference[article_id],
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "selected_fixed_window": validation_result["selected"],
                "long_test_articles": len(test_long),
                "long_test_characters": unique_chars,
                "runtime_rows": runtime_rows,
                "serial_vs_fixed_batch1_agreement": serial_fixed_agreement,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
