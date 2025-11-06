#!/usr/bin/env python3
"""Remove samples that share the same id and duplicate reasoning content."""

import argparse
import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from tabulate import tabulate
except ImportError as exc:  # pragma: no cover - tabulate required at runtime
    raise SystemExit(
        "Missing dependency 'tabulate'. Install it with `pip install tabulate`."
    ) from exc

DATASET_CHOICES = ("gsm8k", "logiqa", "compmath")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = BASE_DIR / "new_dataset" / "2_fbt"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "new_dataset" / "3_fbd"
DEFAULT_OUTPUT_SUFFIX = "_dedup"

WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class DedupOptions:
    collapse_whitespace: bool
    casefold: bool
    overlap_threshold: float


@dataclass
class Record:
    line: str
    active: bool
    sample_id: Optional[str] = None
    tokens: List[str] = field(default_factory=list)
    token_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter JSONL files to drop samples that have duplicate id and reasoning."
        )
    )
    parser.add_argument("dataset", choices=DATASET_CHOICES, help="Dataset name.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing dataset sub-folders with source JSONL files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where deduplicated files will be written.",
    )
    parser.add_argument(
        "--suffix",
        default=DEFAULT_OUTPUT_SUFFIX,
        help="Suffix appended to output file names (default: '_dedup').",
    )
    parser.add_argument(
        "--collapse-whitespace",
        action="store_true",
        default=True,
        help="Treat reasonings that only differ in whitespace as duplicates.",
    )
    parser.add_argument(
        "--case-insensitive",
        action="store_true",
        default=True,
        help="Ignore case differences when comparing reasoning text.",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.75,
        help=(
            "Drop the shorter sample among matching ids when the longest common contiguous "
            "token span covers at least this fraction of the shorter sample. Value must be "
            "between 0 and 1."
        ),
    )
    return parser.parse_args()


def derive_output_path(source: Path, output_dir: Path, suffix: str) -> Path:
    filename = f"{source.stem}{suffix}{source.suffix or '.jsonl'}"
    return output_dir / filename


def should_consider(record: Dict[str, Any]) -> Tuple[bool, str, str]:
    if "id" not in record or "reasoning" not in record:
        return False, "", ""

    sample_id_raw = record["id"]
    reasoning_raw = record["reasoning"]
    if sample_id_raw is None or reasoning_raw is None:
        return False, "", ""

    sample_id = str(sample_id_raw).strip()
    reasoning = str(reasoning_raw).strip()
    if not sample_id or not reasoning:
        return False, "", ""

    return True, sample_id, reasoning


def normalize_reasoning_text(
    reasoning: str, collapse_whitespace: bool, casefold: bool
) -> str:
    text = reasoning
    if collapse_whitespace:
        text = WHITESPACE_RE.sub(" ", text)
    text = text.strip()
    if casefold:
        text = text.casefold()
    return text


def longest_common_span_ratio(tokens_a: List[str], tokens_b: List[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0

    len_a = len(tokens_a)
    len_b = len(tokens_b)
    prev = [0] * (len_b + 1)
    max_span = 0

    for i in range(1, len_a + 1):
        curr = [0] * (len_b + 1)
        token_a = tokens_a[i - 1]
        for j in range(1, len_b + 1):
            if token_a == tokens_b[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > max_span:
                    max_span = curr[j]
        prev = curr

    shorter_len = min(len_a, len_b)
    return (max_span / shorter_len) if shorter_len else 0.0


def deduplicate_file(
    input_path: Path, output_path: Path, options: DedupOptions
) -> Tuple[int, int, int]:
    seen: Dict[str, List[Record]] = {}
    records: List[Record] = []
    total = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue
            total += 1

            newline = raw_line if raw_line.endswith("\n") else raw_line + "\n"

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                records.append(Record(line=newline, active=True))
                continue

            consider, sample_id, reasoning = should_consider(sample)
            serialized = json.dumps(sample, ensure_ascii=False) + "\n"
            if not consider:
                records.append(Record(line=serialized, active=True))
                continue

            normalized = normalize_reasoning_text(
                reasoning,
                collapse_whitespace=options.collapse_whitespace,
                casefold=options.casefold,
            )
            tokens = normalized.split()

            record = Record(
                line=serialized,
                active=True,
                sample_id=sample_id,
                tokens=tokens,
                token_count=len(tokens),
            )
            records.append(record)

            seen_entries = seen.get(sample_id, [])
            active_entries = [entry for entry in seen_entries if entry.active]

            remove_candidate = False
            for existing in list(active_entries):
                span_ratio = longest_common_span_ratio(record.tokens, existing.tokens)
                if span_ratio >= options.overlap_threshold:
                    if record.token_count <= existing.token_count:
                        record.active = False
                        remove_candidate = True
                        break
                    existing.active = False
                    active_entries.remove(existing)

            if remove_candidate:
                seen[sample_id] = active_entries
                continue

            active_entries.append(record)
            seen[sample_id] = active_entries

    kept = sum(1 for record in records if record.active)
    dropped = total - kept

    with output_path.open("w", encoding="utf-8") as dst:
        for record in records:
            if record.active:
                dst.write(record.line)

    return total, kept, dropped


def process_dataset(
    dataset: str,
    input_root: Path,
    output_root: Path,
    suffix: str,
    options: DedupOptions,
) -> None:
    input_dir = input_root / dataset
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.jsonl"))
    if not input_files:
        print(f"[INFO] No JSONL files found in {input_dir}; nothing to do.")
        return

    per_file_rows = []
    per_file_paths = []
    processed_files = 0
    total_samples = 0
    total_kept = 0

    for input_path in input_files:
        output_path = derive_output_path(input_path, output_dir, suffix)
        total, kept, dropped = deduplicate_file(input_path, output_path, options)
        keep_rate = (kept / total * 100) if total else 0.0
        per_file_rows.append(
            [
                input_path.name,
                total,
                kept,
                dropped,
                f"{keep_rate:.1f}%" if total else "n/a",
            ]
        )
        per_file_paths.append((input_path.name, str(output_path.resolve())))

        processed_files += 1
        total_samples += total
        total_kept += kept

    if processed_files == 0:
        print(f"[{dataset}] No files processed. Nothing to do.")
        return

    total_dropped = total_samples - total_kept
    headers = ["File", "Total", "Kept", "Dropped", "Keep Rate"]
    print(f"[{dataset}] Per-file stats:")
    print(tabulate(per_file_rows, headers=headers, tablefmt="grid"))

    print(f"[{dataset}] Output paths:")
    for filename, output_path in per_file_paths:
        print(f"  {filename}: filtered -> {output_path}")

    keep_rate = (total_kept / total_samples * 100) if total_samples else 0.0
    print(
        f"[{dataset}] Summary: processed {processed_files} files, "
        f"kept {total_kept} samples, dropped {total_dropped} samples, "
        f"total {total_samples} samples, keep_rate={keep_rate:.1f}%."
    )


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.overlap_threshold <= 1.0):
        raise ValueError("--overlap-threshold must be between 0 and 1.")

    options = DedupOptions(
        collapse_whitespace=args.collapse_whitespace,
        casefold=args.case_insensitive,
        overlap_threshold=args.overlap_threshold,
    )
    process_dataset(
        args.dataset, args.input_root, args.output_root, args.suffix, options
    )


if __name__ == "__main__":
    main()
