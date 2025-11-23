from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_ORDER = (
    "combined_grpo_train_low.jsonl",
    "combined_grpo_train_medium.jsonl",
    "combined_grpo_train_high.jsonl",
)


def combine_jsonl_files(sources: Iterable[Path], destination: Path) -> None:
    """Concatenate multiple JSONL files into a single JSONL file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as dest_f:
        for source in sources:
            if not source.is_file():
                raise FileNotFoundError(f"{source} does not exist")
            with source.open("r", encoding="utf-8") as src_f:
                for line in src_f:
                    dest_f.write(line)


def main(files: Sequence[str], output: str) -> None:
    base_dir = Path(__file__).resolve().parent
    source_paths = [base_dir / fname for fname in files]
    destination_path = base_dir / output
    combine_jsonl_files(source_paths, destination_path)
    print(f"Combined {len(source_paths)} files into {destination_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenate GRPO train blocks in low > medium > high order."
    )
    parser.add_argument(
        "--order",
        nargs="+",
        default=DEFAULT_ORDER,
        help="Paths relative to this script that should be concatenated in order.",
    )
    parser.add_argument(
        "--output",
        default="combined_grpo_train.jsonl",
        help="Output filename inside the same directory.",
    )

    args = parser.parse_args()
    main(args.order, args.output)
