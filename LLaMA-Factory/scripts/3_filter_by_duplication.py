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

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover - tqdm required at runtime
    raise SystemExit(
        "Missing dependency 'tqdm'. Install it with `pip install tqdm`."
    ) from exc

DATASET_CHOICES = ("gsm8k", "logiqa", "compmath")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = BASE_DIR / "new_dataset" / "2_fbt"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "new_dataset" / "3_fbd"
DEFAULT_OUTPUT_SUFFIX = "_dedup"

WHITESPACE_RE = re.compile(r"\s+")
MAX_TOKENS_FOR_OVERLAP = 256
MAX_MISMATCH_TOKENS = 4


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
    data: Optional[Dict[str, Any]] = None
    position: int = -1
    normalized_reasoning: str = ""


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
        default=0.5,
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


def longest_common_span_ratio(
    tokens_a: List[str], tokens_b: List[str], mismatch_buffer: int = 0
) -> float:
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
    if not shorter_len:
        return 0.0

    adjusted_span = min(shorter_len, max_span + max(0, mismatch_buffer))
    return adjusted_span / shorter_len


def deduplicate_file(
    input_path: Path,
    output_path: Path,
    rejected_path: Path,
    options: DedupOptions,
    show_progress: bool = True,
) -> Tuple[int, int, int, int]:
    seen: Dict[str, List[Record]] = {}
    records: List[Record] = []
    total = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_records: List[Dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as src:
        iterator = (
            tqdm(src, desc=f"{input_path.name}", unit="lines", leave=False)
            if show_progress
            else src
        )
        for raw_line in iterator:
            line = raw_line.strip()
            if not line:
                continue
            total += 1

            newline = raw_line if raw_line.endswith("\n") else raw_line + "\n"

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                record = Record(line=newline, active=True)
                records.append(record)
                record.position = len(records) - 1
                continue

            consider, sample_id, reasoning = should_consider(sample)
            serialized = json.dumps(sample, ensure_ascii=False) + "\n"
            if not consider:
                record = Record(line=serialized, active=True, data=sample)
                records.append(record)
                record.position = len(records) - 1
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
                data=sample,
                normalized_reasoning=normalized,
            )
            records.append(record)
            record.position = len(records) - 1

            seen_entries = seen.get(sample_id, [])
            active_entries = [entry for entry in seen_entries if entry.active]

            remove_candidate = False
            for existing in list(active_entries):
                if (
                    record.normalized_reasoning
                    and record.normalized_reasoning == existing.normalized_reasoning
                ):
                    record.active = False
                    rejected_records.append(
                        {
                            "sample_id": sample_id,
                            "kept_index": existing.position,
                            "dropped_index": record.position,
                            "span_ratio": 1.0,
                            "kept": existing.data,
                            "dropped": record.data,
                        }
                    )
                    remove_candidate = True
                    break

                truncated_new = record.tokens[:MAX_TOKENS_FOR_OVERLAP]
                truncated_existing = existing.tokens[:MAX_TOKENS_FOR_OVERLAP]
                span_ratio = longest_common_span_ratio(
                    truncated_new,
                    truncated_existing,
                    mismatch_buffer=MAX_MISMATCH_TOKENS,
                )
                if span_ratio >= options.overlap_threshold:
                    if record.token_count <= existing.token_count:
                        record.active = False
                        rejected_records.append(
                            {
                                "sample_id": sample_id,
                                "kept_index": existing.position,
                                "dropped_index": record.position,
                                "span_ratio": span_ratio,
                                "kept": existing.data,
                                "dropped": record.data,
                            }
                        )
                        remove_candidate = True
                        break
                    existing.active = False
                    active_entries.remove(existing)
                    rejected_records.append(
                        {
                            "sample_id": sample_id,
                            "kept_index": record.position,
                            "dropped_index": existing.position,
                            "span_ratio": span_ratio,
                            "kept": record.data,
                            "dropped": existing.data,
                        }
                    )

            if remove_candidate:
                seen[sample_id] = active_entries
                continue

            active_entries.append(record)
            seen[sample_id] = active_entries

        if show_progress and hasattr(iterator, "close"):
            iterator.close()

    kept = sum(1 for record in records if record.active)
    dropped = total - kept

    with output_path.open("w", encoding="utf-8") as dst:
        for record in records:
            if record.active:
                dst.write(record.line)

    if rejected_records:
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        with rejected_path.open("w", encoding="utf-8") as rej_dst:
            for item in sorted(rejected_records, key=lambda entry: entry["dropped_index"]):
                payload: Dict[str, Any] = {
                    "sample_id": item["sample_id"],
                    "kept_index": item["kept_index"],
                    "dropped_index": item["dropped_index"],
                    "span_ratio": item["span_ratio"],
                }
                if item.get("kept") is not None:
                    payload["kept"] = item["kept"]
                if item.get("dropped") is not None:
                    payload["dropped"] = item["dropped"]
                rej_dst.write(json.dumps(payload, ensure_ascii=False) + "\n")
    elif rejected_path.exists():
        rejected_path.unlink()

    return total, kept, dropped, len(rejected_records)


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
    rejected_dir = output_root / "rejected" / dataset

    input_files = sorted(input_dir.glob("*.jsonl"))
    if not input_files:
        print(f"[INFO] No JSONL files found in {input_dir}; nothing to do.")
        return

    results: List[Tuple[int, str, str, str, int, int, int, int]] = []
    for index, input_path in enumerate(input_files):
        output_path = derive_output_path(input_path, output_dir, suffix)
        rejected_path = rejected_dir / f"{input_path.stem}{suffix}_rejected.jsonl"
        total, kept, dropped, rejected = deduplicate_file(
            input_path, output_path, rejected_path, options, show_progress=True
        )
        results.append(
            (
                index,
                input_path.name,
                str(output_path.resolve()),
                str(rejected_path.resolve()),
                total,
                kept,
                dropped,
                rejected,
            )
        )

    if not results:
        print(f"[{dataset}] No files processed. Nothing to do.")
        return

    results.sort(key=lambda item: item[0])

    per_file_rows = []
    per_file_paths = []
    total_samples = 0
    total_kept = 0
    total_rejected = 0

    for _, filename, output_path, rejected_path, total, kept, dropped, rejected in results:
        keep_rate = (kept / total * 100) if total else 0.0
        per_file_rows.append(
            [
                filename,
                total,
                kept,
                dropped,
                # rejected,
                f"{keep_rate:.1f}%" if total else "n/a",
            ]
        )
        per_file_paths.append((filename, output_path, rejected_path, rejected))
        total_samples += total
        total_kept += kept
        total_rejected += rejected

    processed_files = len(results)
    total_dropped = total_samples - total_kept
    headers = ["File", "Total", "Kept", "Rejected", "Keep Rate"]
    print(f"[{dataset}] Per-file stats:")
    print(tabulate(per_file_rows, headers=headers, tablefmt="grid"))

    print(f"[{dataset}] Output paths:")
    for filename, output_path, rejected_path, rejected in per_file_paths:
        print(f"  {filename}: filtered -> {output_path}")
        if rejected:
            print(f"           rejected -> {rejected_path}")

    keep_rate = (total_kept / total_samples * 100) if total_samples else 0.0
    print(
        f"[{dataset}] Summary: processed {processed_files} files, "
        f"kept {total_kept} samples, dropped {total_dropped} samples, "
        f"total {total_samples} samples, overlap_rejected={total_rejected}, "
        f"keep_rate={keep_rate:.1f}%.",
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
        args.dataset,
        args.input_root,
        args.output_root,
        args.suffix,
        options,
    )


if __name__ == "__main__":
    main()
