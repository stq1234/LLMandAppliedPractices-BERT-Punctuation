"""Restore punctuation in a UTF-8 text file with the local fine-tuned model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, BertForTokenClassification

from benchmark_long import collate_plan, fuse_probabilities, make_plan
from train import LABELS, ROOT


CHECKPOINT_DIR = ROOT / "checkpoints" / "best_model"
WINDOW_SIZE_CHARS = 256
STRIDE_CHARS = 128
MAX_BATCH_SIZE = 16


def infer_text(
    text: str,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[str, dict[str, Any]]:
    if batch_size < 1 or batch_size > MAX_BATCH_SIZE:
        raise ValueError(f"--batch-size must be between 1 and {MAX_BATCH_SIZE}")
    if not text:
        return "", {
            "input_characters": 0,
            "window_count": 0,
            "window_size_chars": WINDOW_SIZE_CHARS,
            "stride_chars": STRIDE_CHARS,
            "batch_size": batch_size,
        }

    article = {"id": "input", "text": text}
    windows, char_ids_by_article = make_plan(
        [article], tokenizer, WINDOW_SIZE_CHARS, STRIDE_CHARS
    )
    if len(char_ids_by_article["input"]) != len(text):
        raise AssertionError("Tokenizer character IDs do not preserve input alignment")

    probability_rows: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, len(windows), batch_size):
            group = windows[batch_start : batch_start + batch_size]
            cpu_batch = collate_plan(group, int(tokenizer.pad_token_id))
            input_ids = cpu_batch["input_ids"].to(device)
            attention_mask = cpu_batch["attention_mask"].to(device)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            # Shared window offsets refer to characters; logits position 0 is
            # [CLS], so remove it before shared overlap-probability fusion.
            probabilities = torch.softmax(logits.float()[:, 1:, :], dim=-1)
            probability_rows.extend(probabilities.cpu().numpy())

    predictions = fuse_probabilities([article], windows, probability_rows)["input"]
    if len(predictions) != len(text):
        raise AssertionError("The window plan did not return one label per input character")

    original_characters: list[str] = []
    restored_parts: list[str] = []
    for character, label_id in zip(text, predictions, strict=True):
        original_characters.append(character)
        restored_parts.append(character)
        if label_id:
            restored_parts.append(LABELS[int(label_id)])
    if "".join(original_characters) != text:
        raise AssertionError("At least one original input character was lost or reordered")
    restored_text = "".join(restored_parts)

    return restored_text, {
        "input_characters": len(text),
        "window_count": len(windows),
        "window_size_chars": WINDOW_SIZE_CHARS,
        "stride_chars": STRIDE_CHARS,
        "batch_size": batch_size,
        "device": str(device),
        "unknown_characters": sum(token_id == tokenizer.unk_token_id for token_id in char_ids_by_article["input"]),
        "original_characters_preserved_exactly": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="input UTF-8 text file")
    parser.add_argument("--output", required=True, type=Path, help="restored UTF-8 text file")
    parser.add_argument("--batch-size", type=int, default=8, help="inference windows per batch (1–16)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.input.read_text(encoding="utf-8")
    if args.batch_size < 1 or args.batch_size > MAX_BATCH_SIZE:
        raise SystemExit(f"--batch-size must be between 1 and {MAX_BATCH_SIZE}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    if not text:
        restored_text = ""
        stats = {
            "input_characters": 0,
            "window_count": 0,
            "window_size_chars": WINDOW_SIZE_CHARS,
            "stride_chars": STRIDE_CHARS,
            "batch_size": args.batch_size,
            "device": str(device),
            "unknown_characters": 0,
            "original_characters_preserved_exactly": True,
        }
    else:
        if not CHECKPOINT_DIR.is_dir():
            raise FileNotFoundError(
                f"Local checkpoint was not found at {CHECKPOINT_DIR}; train the model first"
            )
        tokenizer = AutoTokenizer.from_pretrained(
            CHECKPOINT_DIR, use_fast=False, local_files_only=True
        )
        model = BertForTokenClassification.from_pretrained(
            CHECKPOINT_DIR, local_files_only=True
        ).to(device)
        if int(model.config.num_labels) != len(LABELS):
            raise ValueError(
                f"Checkpoint has {model.config.num_labels} classes; expected {len(LABELS)}"
            )
        restored_text, stats = infer_text(
            text, model, tokenizer, device, args.batch_size
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(restored_text, encoding="utf-8", newline="")
    summary = {
        **stats,
        "input": str(args.input),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(restored_text.encode("utf-8")).hexdigest(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
