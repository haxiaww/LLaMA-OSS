#!/usr/bin/env python3
"""
Filter only the "medium" entries from a JSONL dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(
        description=(
            "Create a JSONL file that contains only the samples whose "
            'mode is "medium".'
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=Path("data/combined_grpo_train.jsonl"),
        type=Path,
        help="Path to the original combined JSONL file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to the filtered JSONL output file. Defaults to <input>_medium.jsonl",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path.with_name(
        f"{input_path.stem}_medium{input_path.suffix or '.jsonl'}"
    )
    return input_path, output_path


def filter_medium_entries(input_path: Path, output_path: Path) -> None:
    total = 0
    kept = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue

            total += 1
            sample = json.loads(line)

            if sample.get("mode") != "medium":
                continue

            json.dump(sample, dst, ensure_ascii=False)
            dst.write("\n")
            kept += 1

    print(f"Wrote {kept} of {total} entries to {output_path}")


def main() -> None:
    input_path, output_path = parse_args()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    filter_medium_entries(input_path, output_path)


if __name__ == "__main__":
    main()
