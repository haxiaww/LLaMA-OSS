#!/usr/bin/env python3
"""Filter JSONL files so each id keeps a single sample near the mean token count."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DATASET_CHOICES = ("gsm8k", "logiqa", "compmath")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = BASE_DIR / "new_dataset" / "3_fbd"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "new_dataset" / "4_fbi"
DEFAULT_OUTPUT_SUFFIX = "_byid"


@dataclass
class Record:
    line: str
    sample_id: Optional[str]
    value: Optional[float]
    position: int
    data: Optional[Dict[str, Any]]
    active: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collapse samples that share the same id by keeping the entry whose "
            "reasoning_tokens (or selected numeric field) is closest to the group mean. "
            "All dropped samples are written to a rejected file."
        )
    )
    parser.add_argument("dataset", choices=DATASET_CHOICES, help="Dataset name.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing JSONL files to filter.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory that will store filtered files and rejected samples.",
    )
    parser.add_argument(
        "--field",
        default="reasoning_tokens",
        help="Numeric field used to determine which sample to keep (default: reasoning_tokens).",
    )
    parser.add_argument(
        "--id-field",
        default="id",
        help="Field that uniquely identifies a sample/question (default: id).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many JSONL files (useful for quick inspection).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing filtered/rejected files.",
    )
    return parser.parse_args()


def read_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def normalize_id(raw_id: Any) -> Optional[str]:
    if raw_id is None:
        return None
    text = str(raw_id).strip()
    return text or None


def derive_output_path(source: Path, output_dir: Path, suffix: str) -> Path:
    filename = f"{source.stem}{suffix}{source.suffix or '.jsonl'}"
    return output_dir / filename


def read_records(path: Path, id_field: str, value_field: str) -> List[Record]:
    records: List[Record] = []
    with path.open("r", encoding="utf-8") as src:
        for raw_line in src:
            serialized = raw_line if raw_line.endswith("\n") else raw_line + "\n"
            position = len(records)
            try:
                sample = json.loads(raw_line)
            except json.JSONDecodeError:
                records.append(
                    Record(
                        line=serialized,
                        sample_id=None,
                        value=None,
                        position=position,
                        data=None,
                    )
                )
                continue

            sample_id = normalize_id(sample.get(id_field))
            value = read_numeric(sample.get(value_field))
            records.append(
                Record(
                    line=serialized,
                    sample_id=sample_id,
                    value=value,
                    position=position,
                    data=sample,
                )
            )
    return records


def summarize_records(records: List[Record]) -> Dict[str, Any]:
    total = len(records)
    with_ids = [record for record in records if record.sample_id]
    unique_ids = len({record.sample_id for record in with_ids})

    id_counts: Dict[str, int] = {}
    for record in with_ids:
        id_counts[record.sample_id or ""] = id_counts.get(record.sample_id or "", 0) + 1

    multi_id_count = sum(1 for count in id_counts.values() if count > 1)
    duplicate_records = sum(max(0, count - 1) for count in id_counts.values() if count > 1)

    valued = [record.value for record in with_ids if record.value is not None]
    value_mean = statistics.fmean(valued) if valued else None
    value_std = statistics.pstdev(valued) if len(valued) > 1 else None

    return {
        "total": total,
        "with_ids": len(with_ids),
        "unique_ids": unique_ids,
        "multi_ids": multi_id_count,
        "duplicate_records": duplicate_records,
        "value_mean": value_mean,
        "value_std": value_std,
    }


def keep_records_by_mean(records: List[Record]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Record]] = {}
    for record in records:
        if record.sample_id is None:
            continue
        groups.setdefault(record.sample_id, []).append(record)

    rejected_entries: List[Dict[str, Any]] = []

    for sample_id, group in groups.items():
        active_group = [record for record in group if record.active]
        if len(active_group) <= 1:
            continue

        valued_records = [record for record in active_group if record.value is not None]
        group_mean: Optional[float]
        if valued_records:
            group_mean = statistics.fmean(record.value for record in valued_records)
            keep_pool = valued_records
        else:
            group_mean = None
            keep_pool = active_group

        def sort_key(record: Record) -> Tuple[float, int]:
            if record.value is None or group_mean is None:
                return (0.0, record.position)
            return (abs(record.value - group_mean), record.position)

        keep_record = min(keep_pool, key=sort_key)

        for record in active_group:
            if record is keep_record:
                continue
            record.active = False
            rejected_entries.append(
                {
                    "sample_id": sample_id,
                    "kept_index": keep_record.position,
                    "dropped_index": record.position,
                    "group_mean": group_mean,
                    "kept_value": keep_record.value,
                    "dropped_value": record.value,
                    "kept": keep_record.data,
                    "dropped": record.data,
                }
            )

    return rejected_entries


def write_filtered(records: List[Record], destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as dst:
        for record in records:
            if record.active:
                dst.write(record.line)
                count += 1
    return count


def write_rejected(rejected: List[Dict[str, Any]], destination: Path) -> int:
    if not rejected:
        if destination.exists():
            destination.unlink()
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    rejected.sort(key=lambda item: item["dropped_index"])
    with destination.open("w", encoding="utf-8") as dst:
        for entry in rejected:
            dst.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return len(rejected)


def process_dataset(
    dataset: str,
    input_root: Path,
    output_root: Path,
    id_field: str,
    value_field: str,
    limit: Optional[int],
    dry_run: bool,
) -> None:
    input_dir = input_root / dataset
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = output_root / dataset
    rejected_dir = output_root / "rejected" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.jsonl"))
    if limit is not None:
        files = files[:limit]
    if not files:
        print(f"[{dataset}] No input files found in {input_dir}.")
        return

    summary_rows = []
    for index, input_path in enumerate(files, start=1):
        print(f"[INFO] ({index}/{len(files)}) Processing {input_path.name}")
        records = read_records(input_path, id_field=id_field, value_field=value_field)
        stats = summarize_records(records)
        rejected = keep_records_by_mean(records)

        output_path = derive_output_path(input_path, output_dir, DEFAULT_OUTPUT_SUFFIX)
        rejected_path = (
            rejected_dir / f"{input_path.stem}{DEFAULT_OUTPUT_SUFFIX}_rejected.jsonl"
        )

        if dry_run:
            kept_count = sum(1 for record in records if record.active)
            rejected_count = len(rejected)
        else:
            kept_count = write_filtered(records, output_path)
            rejected_count = write_rejected(rejected, rejected_path)

        summary_rows.append(
            {
                "file": input_path.name,
                "total": len(records),
                "kept": kept_count,
                "dropped": len(records) - kept_count,
                "rejected": rejected_count,
                "output": str(output_path) if not dry_run else "(dry run)",
            }
        )

        print(
            f"  - kept={kept_count}, dropped={len(records) - kept_count}, "
            f"rejected={rejected_count}"
        )
        mean_text = (
            f"{stats['value_mean']:.2f}" if stats["value_mean"] is not None else "n/a"
        )
        std_text = (
            f"{stats['value_std']:.2f}" if stats["value_std"] is not None else "n/a"
        )
        print(
            "  - stats:"
            f" unique_ids={stats['unique_ids']} multi_ids={stats['multi_ids']} "
            f"duplicate_records={stats['duplicate_records']} mean_value={mean_text} "
            f"std_value={std_text}"
        )

    print(f"\n[{dataset}] Summary:")
    for row in summary_rows:
        print(
            f"  {row['file']}: total={row['total']} kept={row['kept']} "
            f"dropped={row['dropped']} rejected={row['rejected']} -> {row['output']}"
        )


def main() -> None:
    args = parse_args()
    process_dataset(
        dataset=args.dataset,
        input_root=args.input_root,
        output_root=args.output_root,
        id_field=args.id_field,
        value_field=args.field,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
