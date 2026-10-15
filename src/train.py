"""Fine-tune Chinese BERT for character-level punctuation restoration."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForTokenClassification, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "google-bert/bert-base-chinese"
MODEL_REVISION = "8f23c25b06e129b6c986331a13d8d025a92cf0ea"
SEED = 20261015
LABELS = ("NONE", "，", "。", "？", "！", "；", "：", "、")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
PUNCTUATION_IDS = tuple(range(1, len(LABELS)))
BOUNDARY_IDS = (2, 3, 4)
MAX_MODEL_TOKENS = 256
CONTENT_WINDOW = MAX_MODEL_TOKENS - 2
EPOCHS = 3
LEARNING_RATE = 2e-5


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass
class Window:
    article_id: str
    start: int
    end: int
    input_ids: list[int]
    labels: list[int]


def encode_articles(
    articles: list[dict[str, Any]], tokenizer: Any
) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, str], dict[str, int]]:
    char_ids: dict[str, list[int]] = {}
    labels: dict[str, list[int]] = {}
    texts: dict[str, str] = {}
    unknown_counts: collections.Counter[str] = collections.Counter()
    for article in articles:
        article_id = str(article["id"])
        text = str(article["text"])
        ids = tokenizer.convert_tokens_to_ids(list(text))
        if len(ids) != len(text) or len(article["labels"]) != len(text):
            raise ValueError(f"Character/ID/label alignment failure for {article_id}")
        char_ids[article_id] = ids
        labels[article_id] = [int(value) for value in article["labels"]]
        texts[article_id] = text
        unknown_counts["characters"] += len(text)
        unknown_counts["unk_characters"] += sum(
            token_id == tokenizer.unk_token_id for token_id in ids
        )
    return char_ids, labels, texts, dict(unknown_counts)


def create_windows(
    articles: list[dict[str, Any]],
    char_ids: dict[str, list[int]],
    article_labels: dict[str, list[int]],
    tokenizer: Any,
    content_window: int = CONTENT_WINDOW,
) -> list[Window]:
    windows: list[Window] = []
    cls_id = int(tokenizer.cls_token_id)
    sep_id = int(tokenizer.sep_token_id)
    for article in articles:
        article_id = str(article["id"])
        ids = char_ids[article_id]
        gold = article_labels[article_id]
        for start in range(0, len(ids), content_window):
            end = min(start + content_window, len(ids))
            windows.append(
                Window(
                    article_id=article_id,
                    start=start,
                    end=end,
                    input_ids=[cls_id, *ids[start:end], sep_id],
                    labels=[-100, *gold[start:end], -100],
                )
            )
        if sum(window.end - window.start for window in windows if window.article_id == article_id) != len(ids):
            raise AssertionError(f"Window coverage mismatch for {article_id}")
    return windows


class WindowDataset(Dataset[Window]):
    def __init__(self, windows: list[Window]):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Window:
        return self.windows[index]


def collate_windows(batch: list[Window], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_length = max(len(window.input_ids) for window in batch)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []
    for window in batch:
        padding = max_length - len(window.input_ids)
        input_ids.append(window.input_ids + [pad_token_id] * padding)
        attention_mask.append([1] * len(window.input_ids) + [0] * padding)
        labels.append(window.labels + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def class_weights(labels: dict[str, list[int]]) -> torch.Tensor:
    counts = np.zeros(len(LABELS), dtype=np.int64)
    for sequence in labels.values():
        counts += np.bincount(sequence, minlength=len(LABELS))
    punctuation_counts = counts[1:]
    median_punctuation = float(np.median(punctuation_counts))
    weights = np.ones(len(LABELS), dtype=np.float32)
    weights[1:] = np.clip(np.sqrt(median_punctuation / punctuation_counts), 0.5, 4.0)
    return torch.tensor(weights, dtype=torch.float32)


def confusion_metrics(confusion: np.ndarray) -> dict[str, Any]:
    total = int(confusion.sum())
    diagonal = np.diag(confusion)
    punct_tp = int(confusion[1:, 1:].diagonal().sum())
    punct_fp = int(confusion[:, 1:].sum()) - punct_tp
    punct_fn = int(confusion[1:, :].sum()) - punct_tp
    punct_precision = punct_tp / (punct_tp + punct_fp) if punct_tp + punct_fp else 0.0
    punct_recall = punct_tp / (punct_tp + punct_fn) if punct_tp + punct_fn else 0.0
    punct_f1 = (
        2 * punct_precision * punct_recall / (punct_precision + punct_recall)
        if punct_precision + punct_recall
        else 0.0
    )
    per_class: dict[str, dict[str, float | int]] = {}
    class_f1s = []
    for label_id in PUNCTUATION_IDS:
        tp = int(confusion[label_id, label_id])
        fp = int(confusion[:, label_id].sum()) - tp
        fn = int(confusion[label_id, :].sum()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_f1s.append(f1)
        per_class[LABELS[label_id]] = {
            "support": int(confusion[label_id, :].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    boundary_tp = int(confusion[np.ix_(BOUNDARY_IDS, BOUNDARY_IDS)].diagonal().sum())
    boundary_fp = int(confusion[:, BOUNDARY_IDS].sum()) - boundary_tp
    boundary_fn = int(confusion[BOUNDARY_IDS, :].sum()) - boundary_tp
    boundary_precision = boundary_tp / (boundary_tp + boundary_fp) if boundary_tp + boundary_fp else 0.0
    boundary_recall = boundary_tp / (boundary_tp + boundary_fn) if boundary_tp + boundary_fn else 0.0
    boundary_f1 = (
        2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
        if boundary_precision + boundary_recall
        else 0.0
    )
    return {
        "characters": total,
        "token_accuracy": float(diagonal.sum() / total) if total else 0.0,
        "always_none_accuracy": float(confusion[0, :].sum() / total) if total else 0.0,
        "none_recall": float(confusion[0, 0] / confusion[0, :].sum()) if confusion[0, :].sum() else 0.0,
        "punctuation_micro_precision": punct_precision,
        "punctuation_micro_recall": punct_recall,
        "punctuation_micro_f1": punct_f1,
        "punctuation_macro_f1": float(np.mean(class_f1s)),
        "sentence_boundary_precision": boundary_precision,
        "sentence_boundary_recall": boundary_recall,
        "sentence_boundary_f1": boundary_f1,
        "per_class": per_class,
    }


def evaluate(
    model: torch.nn.Module,
    windows: list[Window],
    articles: list[dict[str, Any]],
    texts: dict[str, str],
    article_labels: dict[str, list[int]],
    pad_token_id: int,
    batch_size: int,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[dict[str, Any], dict[str, list[int]], np.ndarray]:
    model.eval()
    loader = DataLoader(
        WindowDataset(windows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_windows(batch, pad_token_id),
        pin_memory=device.type == "cuda",
    )
    predictions = {str(article["id"]): [-1] * len(texts[str(article["id"])]) for article in articles}
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    window_offset = 0
    with torch.inference_mode():
        for batch in loader:
            current_windows = windows[window_offset : window_offset + len(batch["input_ids"])]
            window_offset += len(current_windows)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predicted = logits.argmax(dim=-1).cpu().numpy()
            for row, window in enumerate(current_windows):
                predicted_content = predicted[row, 1 : 1 + window.end - window.start].tolist()
                predictions[window.article_id][window.start : window.end] = predicted_content
                gold_content = article_labels[window.article_id][window.start : window.end]
                confusion += np.bincount(
                    np.asarray(gold_content, dtype=np.int64) * len(LABELS)
                    + np.asarray(predicted_content, dtype=np.int64),
                    minlength=len(LABELS) * len(LABELS),
                ).reshape(len(LABELS), len(LABELS))
    if any(value < 0 for sequence in predictions.values() for value in sequence):
        raise AssertionError("Evaluation left one or more original characters without a prediction")
    metrics = confusion_metrics(confusion)
    return metrics, predictions, confusion


def run_training_smoke(
    model: BertForTokenClassification,
    windows: list[Window],
    pad_token_id: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    candidates = sorted(windows, key=lambda window: len(window.input_ids), reverse=True)
    chosen: list[Window] = []
    padded_length = min(MAX_MODEL_TOKENS, max(len(window.input_ids) for window in candidates))
    for window in candidates:
        if len(chosen) == batch_size:
            break
        if len(window.input_ids) == padded_length:
            chosen.append(window)
    if len(chosen) < batch_size:
        chosen.extend(candidates[: batch_size - len(chosen)])
    batch = collate_windows(chosen, pad_token_id)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    times: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits.reshape(-1, len(LABELS)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if step:
            times.append(elapsed)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite smoke loss")
    peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "batch_size": batch_size,
        "sequence_length": int(input_ids.shape[1]),
        "optimizer_step_seconds_median_after_warmup": statistics.median(times),
        "peak_allocated_mib": peak_bytes / (1024**2),
        "windows_per_epoch": len(windows),
        "estimated_steps_per_epoch": math.ceil(len(windows) / batch_size),
    }


def save_test_predictions(
    articles: list[dict[str, Any]],
    predictions: dict[str, list[int]],
    article_labels: dict[str, list[int]],
    texts: dict[str, str],
) -> None:
    output = ROOT / "results" / "test_predictions.jsonl"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for article in articles:
            article_id = str(article["id"])
            row = {
                "article_id": article_id,
                "text": texts[article_id],
                "gold_label_ids": article_labels[article_id],
                "predicted_label_ids": predictions[article_id],
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def save_errors(
    articles: list[dict[str, Any]],
    predictions: dict[str, list[int]],
    article_labels: dict[str, list[int]],
    texts: dict[str, str],
    model: torch.nn.Module,
    tokenizer: Any,
    pad_token_id: int,
    device: torch.device,
    batch_size: int,
) -> None:
    # The compact error table is formed from saved predictions; confidence is
    # deliberately omitted because the full character-level prediction file
    # is the auditable record and no calibration claim is made.
    errors_by_gold: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for article in articles:
        article_id = str(article["id"])
        text = texts[article_id]
        gold = article_labels[article_id]
        pred = predictions[article_id]
        for index, (gold_id, pred_id) in enumerate(zip(gold, pred)):
            if gold_id == pred_id:
                continue
            errors_by_gold[gold_id].append(
                {
                    "article_id": article_id,
                    "position": index,
                    "gold_label_id": int(gold_id),
                    "predicted_label_id": int(pred_id),
                    "context": text[max(0, index - 24) : min(len(text), index + 25)],
                    "character": text[index],
                }
            )
    selected = []
    for gold_id in range(len(LABELS)):
        selected.extend(errors_by_gold.get(gold_id, [])[:5])
    selected.sort(key=lambda item: (item["gold_label_id"], item["article_id"], item["position"]))
    out = ROOT / "results" / "error_examples.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for error in selected:
            error["gold_label"] = LABELS[error["gold_label_id"]]
            error["predicted_label"] = LABELS[error["predicted_label_id"]]
            handle.write(json.dumps(error, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true", help="download the model and benchmark four AMP optimizer steps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    if args.batch_size > 8:
        raise ValueError("Training batch size must be <= 8 at length 256 on the assigned GPU")
    if args.gradient_accumulation < 1:
        raise ValueError("gradient accumulation must be >= 1")

    seed_everything(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the requested local GPU training cannot proceed")
    device = torch.device("cuda")
    train_articles = read_jsonl(ROOT / "data" / "train.jsonl")
    val_articles = read_jsonl(ROOT / "data" / "validation.jsonl")
    test_articles = read_jsonl(ROOT / "data" / "test.jsonl")
    split_sets = [{str(row["id"]) for row in subset} for subset in (train_articles, val_articles, test_articles)]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("Article ID leakage across train/validation/test splits")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        use_fast=False,
    )
    id2label = {index: label for index, label in enumerate(LABELS)}
    label2id = {label: index for index, label in enumerate(LABELS)}
    model = BertForTokenClassification.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        use_safetensors=True,
    )
    model.to(device)

    all_articles = train_articles + val_articles + test_articles
    all_char_ids, all_labels, all_texts, unknown_stats = encode_articles(all_articles, tokenizer)
    train_windows = create_windows(train_articles, all_char_ids, all_labels, tokenizer)
    val_windows = create_windows(val_articles, all_char_ids, all_labels, tokenizer)
    test_windows = create_windows(test_articles, all_char_ids, all_labels, tokenizer)
    tokenizer_stats = {
        **unknown_stats,
        "unk_rate": unknown_stats["unk_characters"] / unknown_stats["characters"],
        "method": "direct WordPiece vocabulary lookup for each original Unicode character; [UNK] if absent",
        "special_tokens": {"cls": tokenizer.cls_token_id, "sep": tokenizer.sep_token_id, "pad": tokenizer.pad_token_id},
        "max_model_tokens": MAX_MODEL_TOKENS,
        "content_window": CONTENT_WINDOW,
        "window_policy": "non-overlapping; every original character receives exactly one gold label and one evaluation prediction",
        "window_counts": {"train": len(train_windows), "validation": len(val_windows), "test": len(test_windows)},
    }
    (ROOT / "results" / "tokenizer_audit.json").write_text(
        json.dumps(tokenizer_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.smoke_only:
        result = run_training_smoke(
            model,
            train_windows,
            int(tokenizer.pad_token_id),
            device,
            args.batch_size,
        )
        median_step = float(result["optimizer_step_seconds_median_after_warmup"])
        steps_per_epoch = int(result["estimated_steps_per_epoch"])
        result["estimated_three_epoch_train_minutes"] = median_step * steps_per_epoch * 3 / 60
        result["model"] = MODEL_NAME
        result["model_revision"] = MODEL_REVISION
        result["gpu_name"] = torch.cuda.get_device_name(0)
        (ROOT / "results" / "training_smoke.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    batch_size = args.batch_size
    class_weight_tensor = class_weights({key: all_labels[key] for key in all_labels if key in split_sets[0]}).to(device)
    train_loader = DataLoader(
        WindowDataset(train_windows),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_windows(batch, int(tokenizer.pad_token_id)),
        pin_memory=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    warmup_updates = max(1, int(total_updates * 0.06))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_updates, total_updates)
    scaler = torch.amp.GradScaler("cuda", init_scale=8192.0, enabled=True)
    best_val_f1 = -1.0
    best_epoch = 0
    epoch_rows: list[dict[str, Any]] = []
    model_dir = ROOT / "checkpoints" / "best_model"
    epoch_started_all = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        steps = 0
        skipped_updates = 0
        epoch_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        for batch_index, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, len(LABELS)),
                    labels.reshape(-1),
                    weight=class_weight_tensor,
                    ignore_index=-100,
                )
                scaled_loss = loss / args.gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {batch_index}")
            scaler.scale(scaled_loss).backward()
            loss_sum += float(loss.detach().item())
            should_update = batch_index % args.gradient_accumulation == 0 or batch_index == len(train_loader)
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                    steps += 1
                else:
                    skipped_updates += 1
        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - epoch_started
        val_metrics, _, _ = evaluate(
            model,
            val_windows,
            val_articles,
            all_texts,
            all_labels,
            int(tokenizer.pad_token_id),
            max(batch_size, 8),
            device,
        )
        is_best = val_metrics["punctuation_micro_f1"] > best_val_f1
        if is_best:
            best_val_f1 = float(val_metrics["punctuation_micro_f1"])
            best_epoch = epoch
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, len(train_loader)),
            "updates": steps,
            "amp_skipped_updates": skipped_updates,
            "train_seconds": train_seconds,
            "validation": val_metrics,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        }
        epoch_rows.append(row)
        (ROOT / "results" / "training_metrics.json").write_text(
            json.dumps(
                {
                    "model": MODEL_NAME,
                    "model_revision": MODEL_REVISION,
                    "seed": SEED,
                    "epochs_requested": args.epochs,
                    "max_model_tokens": MAX_MODEL_TOKENS,
                    "content_window": CONTENT_WINDOW,
                    "batch_size": batch_size,
                    "gradient_accumulation": args.gradient_accumulation,
                    "learning_rate": LEARNING_RATE,
                    "class_weights": class_weight_tensor.detach().cpu().tolist(),
                    "selection_metric": "validation punctuation micro-F1 (all seven target labels, excluding NONE)",
                    "best_epoch_so_far": best_epoch,
                    "best_validation_punctuation_micro_f1_so_far": best_val_f1,
                    "epochs": epoch_rows,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if is_best:
            model.save_pretrained(model_dir, safe_serialization=True)
            tokenizer.save_pretrained(model_dir)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": row["train_loss"],
                    "train_minutes": train_seconds / 60,
                    "val_punctuation_micro_f1": val_metrics["punctuation_micro_f1"],
                    "val_punctuation_macro_f1": val_metrics["punctuation_macro_f1"],
                    "val_sentence_boundary_f1": val_metrics["sentence_boundary_f1"],
                    "peak_allocated_mib": row["peak_allocated_mib"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    (ROOT / "results" / "training_metrics.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "seed": SEED,
                "epochs_requested": args.epochs,
                "max_model_tokens": MAX_MODEL_TOKENS,
                "content_window": CONTENT_WINDOW,
                "batch_size": batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "learning_rate": LEARNING_RATE,
                "class_weights": class_weight_tensor.detach().cpu().tolist(),
                "selection_metric": "validation punctuation micro-F1 (all seven target labels, excluding NONE)",
                "best_epoch": best_epoch,
                "best_validation_punctuation_micro_f1": best_val_f1,
                "total_train_minutes": (time.perf_counter() - epoch_started_all) / 60,
                "epochs": epoch_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    best_model = BertForTokenClassification.from_pretrained(model_dir).to(device)
    test_metrics, test_predictions, test_confusion = evaluate(
        best_model,
        test_windows,
        test_articles,
        all_texts,
        all_labels,
        int(tokenizer.pad_token_id),
        max(batch_size, 8),
        device,
    )
    test_output = {
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "best_epoch": best_epoch,
        "selection_metric": "validation punctuation micro-F1 (all seven target labels, excluding NONE)",
        "metrics": test_metrics,
    }
    (ROOT / "results" / "test_metrics.json").write_text(
        json.dumps(test_output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (ROOT / "results" / "test_confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold\\pred", *LABELS])
        for label_id, row in enumerate(test_confusion):
            writer.writerow([LABELS[label_id], *[int(value) for value in row]])
    save_test_predictions(test_articles, test_predictions, all_labels, all_texts)
    save_errors(test_articles, test_predictions, all_labels, all_texts, best_model, tokenizer, int(tokenizer.pad_token_id), device, batch_size)

    checkpoint_hashes = {}
    for path in sorted(model_dir.iterdir()):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            checkpoint_hashes[path.name] = {"bytes": path.stat().st_size, "sha256": digest}
    (ROOT / "results" / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "model_dir": "checkpoints/best_model",
                "base_model": MODEL_NAME,
                "base_model_revision": MODEL_REVISION,
                "best_epoch": best_epoch,
                "validation_punctuation_micro_f1": best_val_f1,
                "files": checkpoint_hashes,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"best_epoch": best_epoch, "test_metrics": test_metrics, "checkpoint_files": checkpoint_hashes}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
