#!/usr/bin/env python3
"""Keep inference records whose ids appear in a reference dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Set, Tuple
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune inference JSONL records whose sample ids are absent from a reference dataset."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the inference JSONL file that contains 'id' and 'generation_index'.",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "competition_math_train_alpaca.jsonl",
        help="Reference dataset whose line order defines valid sample ids (default: competition math train).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Destination for the filtered records. (Default: <input>.filtered.jsonl)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file with the filtered dataset (cannot be combined with --output-file).",
    )
    return parser.parse_args()


def load_dataset_ids(dataset_path: Path) -> Set[int]:
    valid_ids: Set[int] = set()
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} does not exist.")
    with dataset_path.open("r", encoding="utf-8") as src:
        for index, raw_line in enumerate(src):
            if not raw_line.strip():
                continue
            try:
                json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            valid_ids.add(index)
    return valid_ids


def filter_records(
    source_path: Path,
    destination_path: Path,
    valid_ids: Set[int],
) -> Tuple[int, int, int, Set[int]]:
    kept_count = 0
    removed_count = 0
    total_lines = 0
    removed_ids: Set[int] = set()
    with source_path.open("r", encoding="utf-8") as src, destination_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for raw_line in tqdm(src):
            total_lines += 1
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            sample_id = record.get("id")
            try:
                sample_id_int = int(sample_id)
            except (TypeError, ValueError):
                removed_count += 1
                continue
            if sample_id_int in valid_ids:
                dst.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                kept_count += 1
            else:
                removed_count += 1
                removed_ids.add(sample_id_int)
    return total_lines, kept_count, removed_count, removed_ids


def main() -> None:
    args = parse_args()
    if args.output_file and args.in_place:
        raise SystemExit("Cannot use --output-file and --in-place together.")

    valid_ids = load_dataset_ids(args.dataset_file)

    if args.in_place:
        target_output = args.input_file.with_suffix(args.input_file.suffix + ".filtered.tmp")
    else:
        target_output = (
            args.output_file
            if args.output_file
            else args.input_file.with_name(
                f"{args.input_file.stem}.filtered{args.input_file.suffix}"
            )
        )

    total_lines, kept, removed, removed_ids = filter_records(
        args.input_file, target_output, valid_ids
    )

    if args.in_place:
        target_output.replace(args.input_file)

    print(f"Dataset reference: {args.dataset_file} (valid ids: {len(valid_ids)})")
    print(f"Processed {total_lines} lines from {args.input_file}")
    print(f"  kept {kept} records, removed {removed} invalid entries ({len(removed_ids)} unique ids)")
    print(f"Filtered data saved at: {target_output if not args.in_place else args.input_file}")


if __name__ == "__main__":
    main()
