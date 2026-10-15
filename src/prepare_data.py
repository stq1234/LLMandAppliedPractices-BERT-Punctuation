"""Prepare the January People's Daily punctuation restoration corpus."""

from __future__ import annotations

import collections
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "1998-01-2003版-带音.txt"
ARCHIVE_PATH = ROOT.parent / "1998-上半年-带音(1).rar"
RAW_SHA256 = "37b85555df2f72642ff9815ac01f51d4f50024d80ca0e2e8a6be6f93b1038639"
ARCHIVE_BYTES = 12093296
ARCHIVE_SHA256 = "30c4435bcce7d082c2175720a4549114f80656761df95d3c8b20f0ee48e5d7ab"
SEED = 20261015
PUNCTUATION = ("，", "。", "？", "！", "；", "：", "、")
LABELS = ("NONE",) + PUNCTUATION
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
NORMALIZED_TARGETS = {
    ",": "，",
    "。": "。",
    "?": "？",
    "!": "！",
    ";": "；",
    ":": "：",
    "、": "、",
    "∶": "：",
}
LAYOUT_SYMBOLS = frozenset("〓●○△▲")
ANNOTATION_RE = re.compile(r"\{[^{}]*\}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_token(token: str, stats: collections.Counter[str]) -> str:
    before = token
    token = ANNOTATION_RE.sub("", token)
    if token != before:
        stats["phonetic_annotations_removed"] += before.count("{")
    # Targets are mapped after NFKC, which turns full-width comma/question/
    # exclamation/semicolon/colon into ASCII. ∶ is an explicit colon alias.
    return unicodedata.normalize("NFKC", token)


def clean_content(token: str, stats: collections.Counter[str]) -> str:
    kept: list[str] = []
    for index, char in enumerate(token):
        category = unicodedata.category(char)
        if category in {"Cc", "Cf", "Cs", "Cn"}:
            stats[f"layout_{category}_characters_removed"] += 1
            continue
        if char in LAYOUT_SYMBOLS:
            stats[f"layout_symbol_removed:{char}"] += 1
            continue
        if category.startswith("P"):
            if (
                char == "."
                and index > 0
                and index + 1 < len(token)
                and token[index - 1].isdigit()
                and token[index + 1].isdigit()
            ):
                stats["numeric_internal_decimal_points_kept"] += 1
                kept.append(char)
            else:
                stats[f"unsupported_punctuation_removed:{char}"] += 1
            continue
        kept.append(char)
    return "".join(kept)


def normalize_article(
    rows: list[tuple[int, str]], stats: collections.Counter[str]
) -> tuple[str, list[int]]:
    characters: list[str] = []
    labels: list[int] = []
    previous_character_index: int | None = None
    punctuation_pending = False

    for _, body in sorted(rows):
        for raw_token in body.split():
            token = clean_token(raw_token, stats)
            if not token:
                continue

            # A punctuation token is one or more standalone target marks. When
            # several follow one character, the final mark is representable by
            # the single-label classifier and becomes its target.
            if all(char in NORMALIZED_TARGETS for char in token):
                for char in token:
                    if previous_character_index is None:
                        stats["leading_target_marks_without_preceding_character"] += 1
                        continue
                    if punctuation_pending:
                        stats["consecutive_target_marks_collapsed"] += 1
                    labels[previous_character_index] = LABEL_TO_ID[NORMALIZED_TARGETS[char]]
                    punctuation_pending = True
                continue

            token = clean_content(token, stats)
            for char in token:
                characters.append(char)
                labels.append(LABEL_TO_ID["NONE"])
                previous_character_index = len(characters) - 1
                punctuation_pending = False

    text = "".join(characters)
    if len(text) != len(labels):
        raise AssertionError("Normalized character skeleton and labels are misaligned")
    for index, char in enumerate(text):
        if unicodedata.category(char).startswith("P") and not (
            char == "."
            and index > 0
            and index + 1 < len(text)
            and text[index - 1].isdigit()
            and text[index + 1].isdigit()
        ):
            raise AssertionError(f"Non-decimal punctuation survived normalization: {char!r}")
        if char in LAYOUT_SYMBOLS:
            raise AssertionError(f"Layout symbol survived normalization: {char!r}")
    if any(char.isspace() for char in text):
        raise AssertionError("Whitespace/newline survived canonical text construction")
    return text, labels


def main() -> None:
    raw = RAW_PATH.read_bytes()
    raw_hash = sha256(raw)
    if raw_hash != RAW_SHA256:
        raise ValueError(
            f"Raw corpus SHA-256 mismatch: expected {RAW_SHA256}, got {raw_hash}"
        )

    records: dict[str, set[str]] = collections.defaultdict(set)
    record_occurrences: collections.Counter[tuple[str, str]] = collections.Counter()
    malformed_lines: list[int] = []
    nonempty_lines = 0
    for line_number, line in enumerate(raw.decode("gb18030").splitlines(), start=1):
        if not line.strip():
            continue
        nonempty_lines += 1
        match = re.match(r"^(\S+)\s+(.*)$", line)
        if match is None:
            malformed_lines.append(line_number)
            continue
        record_id, body = match.groups()
        records[record_id].add(body)
        record_occurrences[(record_id, body)] += 1

    duplicate_id_copies = nonempty_lines - len(records)
    identical_duplicate_copies = sum(count - 1 for count in record_occurrences.values() if count > 1)
    identical_duplicate_ids = sorted(
        record_id for (record_id, _), count in record_occurrences.items() if count > 1
    )
    conflicting_ids = sorted(record_id for record_id, values in records.items() if len(values) > 1)
    conflicting_articles = sorted({record_id.rsplit("-", 1)[0] for record_id in conflicting_ids})

    article_rows: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    title_records = 0
    source_article_rows: dict[str, set[int]] = collections.defaultdict(set)
    for record_id, values in records.items():
        article_id, row_id = record_id.rsplit("-", 1)
        row_number = int(row_id)
        source_article_rows[article_id].add(row_number)
        if len(values) != 1:
            continue
        if row_number == 1:
            title_records += 1
            continue
        if article_id in conflicting_articles:
            continue
        article_rows[article_id].append((row_number, next(iter(values))))

    stats: collections.Counter[str] = collections.Counter()
    cleaned_articles: list[dict[str, object]] = []
    bodyless_articles = 0
    for article_id, rows in sorted(article_rows.items()):
        text, labels = normalize_article(rows, stats)
        if not text:
            bodyless_articles += 1
            continue
        cleaned_articles.append({"id": article_id, "text": text, "labels": labels})

    # Exact normalized article identity includes the character skeleton and
    # its punctuation targets. Retain the lexicographically first source ID.
    seen_normalized: dict[bytes, str] = {}
    deduplicated: list[dict[str, object]] = []
    for article in sorted(cleaned_articles, key=lambda item: str(item["id"])):
        identity = str(article["text"]).encode("utf-8") + b"\0" + bytes(article["labels"])
        digest = hashlib.sha256(identity).digest()
        if digest in seen_normalized:
            stats["duplicate_normalized_articles_removed"] += 1
            continue
        seen_normalized[digest] = str(article["id"])
        deduplicated.append(article)

    shuffled = sorted(
        deduplicated,
        key=lambda item: hashlib.sha256(
            f"{SEED}\0{item['id']}".encode("utf-8")
        ).hexdigest(),
    )
    n_train = math.floor(len(shuffled) * 0.8)
    n_val = math.floor(len(shuffled) * 0.1)
    splits = {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
    split_id_sets = [{str(example["id"]) for example in examples} for examples in splits.values()]
    if len(set.union(*split_id_sets)) != sum(map(len, split_id_sets)):
        raise AssertionError("Article ID leakage or duplicate ID across dataset splits")

    for split_name, examples in splits.items():
        output = ROOT / "data" / f"{split_name}.jsonl"
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            for example in examples:
                handle.write(
                    json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n"
                )

    split_ids = {name: [str(example["id"]) for example in items] for name, items in splits.items()}
    split_hashes = {
        name: sha256((ROOT / "data" / f"{name}.jsonl").read_bytes()) for name in splits
    }
    total_chars = sum(len(str(example["text"])) for example in deduplicated)
    total_labels = collections.Counter(
        label
        for example in deduplicated
        for label in (LABELS[int(label_id)] for label_id in example["labels"])
    )
    archive_reverified = False
    if ARCHIVE_PATH.exists():
        archive_bytes = ARCHIVE_PATH.read_bytes()
        archive_hash = sha256(archive_bytes)
        if len(archive_bytes) != ARCHIVE_BYTES or archive_hash != ARCHIVE_SHA256:
            raise ValueError("Original archive does not match the recorded source provenance")
        archive_reverified = True

    audit = {
        "source_file": RAW_PATH.name,
        "source_bytes": len(raw),
        "source_sha256": raw_hash,
        "source_archive": ARCHIVE_PATH.name,
        "source_archive_bytes": ARCHIVE_BYTES,
        "source_archive_sha256": ARCHIVE_SHA256,
        "source_archive_reverified_in_current_workspace": archive_reverified,
        "source_encoding": "GB18030",
        "source_nonempty_lines": nonempty_lines,
        "unique_record_ids": len(records),
        "duplicate_record_id_copies": duplicate_id_copies,
        "identical_duplicate_record_copies": identical_duplicate_copies,
        "identical_duplicate_record_ids": identical_duplicate_ids,
        "conflicting_record_ids": conflicting_ids,
        "conflicting_articles_excluded": conflicting_articles,
        "malformed_line_numbers": malformed_lines,
        "title_records_excluded": title_records,
        "source_article_id_count": len(source_article_rows),
        "title_only_articles_excluded": sum(rows == {1} for rows in source_article_rows.values()),
        "articles_missing_title_row": sorted(
            article_id for article_id, rows in source_article_rows.items() if 1 not in rows
        ),
        "articles_with_body_before_dedup": len(cleaned_articles),
        "bodyless_articles_after_normalization_excluded": bodyless_articles,
        "articles_after_normalized_dedup": len(deduplicated),
        "characters_after_normalization": total_chars,
        "label_counts": {label: total_labels[label] for label in LABELS},
        "labels": list(LABELS),
        "split_seed": SEED,
        "split_article_counts": {name: len(items) for name, items in splits.items()},
        "split_character_counts": {
            name: sum(len(str(example["text"])) for example in items)
            for name, items in splits.items()
        },
        "split_jsonl_sha256": split_hashes,
        "split_article_ids": split_ids,
        "normalization": {
            "title_row": "row 001 excluded",
            "body_rows": "sorted by numeric row ID and concatenated within each article; line breaks are removed",
            "phonetic_annotations": "all brace annotations such as {shang5} removed",
            "layout_symbols": "Unicode control/format/surrogate/unassigned characters removed",
            "unicode": "NFKC; compatibility target forms are mapped before punctuation filtering",
            "standalone_targets": "，。？！；：、 plus ASCII compatibility forms and standalone ∶ (colon) become the label on the immediately preceding character",
            "consecutive_targets": "the final target mark overwrites earlier marks on that preceding character",
            "unsupported_punctuation": "removed from the input character skeleton and counted by character",
            "internal_punctuation": "removed, except a period between digits inside the same token; such decimal points remain input characters with NONE labels",
            "layout_symbols": "〓, ●, ○, △, ▲ explicitly removed and counted; Unicode control/format/surrogate/unassigned characters are also removed",
            "deduplication": "SHA-256 of normalized character skeleton plus label-byte sequence; first sorted article ID retained",
        },
        "normalization_counters": dict(stats),
    }
    (ROOT / "results" / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "source_sha256": raw_hash,
                "records": len(records),
                "duplicate_record_id_copies": duplicate_id_copies,
                "conflicting_ids": conflicting_ids,
                "conflicting_articles_excluded": conflicting_articles,
                "articles_after_dedup": len(deduplicated),
                "characters": total_chars,
                "split_article_counts": audit["split_article_counts"],
                "split_character_counts": audit["split_character_counts"],
                "label_counts": audit["label_counts"],
                "normalization_counters": audit["normalization_counters"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
