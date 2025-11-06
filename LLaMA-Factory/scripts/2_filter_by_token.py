#!/usr/bin/env python3
"""Filter dataset JSONL files by reasoning token thresholds."""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

try:
    from tabulate import tabulate
except ImportError as exc:  # pragma: no cover - tabulate required at runtime
    raise SystemExit(
        "Missing dependency 'tabulate'. Install it with `pip install tabulate`."
    ) from exc

DATASET_CHOICES = ("gsm8k", "logiqa", "compmath")
MODE_CHOICES = ("high", "medium", "low")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = BASE_DIR / "new_dataset" / "1_fbl"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "new_dataset" / "2_fbt"
BOXED_ONLY_PATTERN = re.compile(r"^\s*\\boxed\{([^}]+)\}\s*$")


def is_valid_predict(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return BOXED_ONLY_PATTERN.match(value) is not None


def get_float_field(record: Dict[str, Any], key: str) -> Optional[float]:
    if key not in record:
        return None
    try:
        return float(record[key])
    except (TypeError, ValueError):
        return None


def should_keep(
    record: Dict[str, Any],
    min_reasoning_threshold: float,
    max_output_threshold: Optional[float],
) -> bool:
    reasoning_value = get_float_field(record, "reasoning_tokens")
    if reasoning_value is None or reasoning_value < min_reasoning_threshold:
        return False

    if max_output_threshold is not None:
        output_value = get_float_field(record, "output_tokens")
        if output_value is None or output_value > max_output_threshold:
            return False

    if not is_valid_predict(record.get("predict")):
        return False
    return True


def format_threshold(threshold: float) -> str:
    text = f"{threshold:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def ensure_parent_dir(path: Path) -> None:
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def filter_file(
    input_path: Path,
    output_path: Path,
    min_threshold: float,
    max_threshold: Optional[float],
) -> Tuple[int, int]:
    ensure_parent_dir(output_path)
    total = 0
    kept = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if should_keep(obj, min_threshold, max_threshold):
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
    return total, kept


def build_threshold_tag(min_threshold: float, max_threshold: Optional[float]) -> str:
    parts = [f"min-{format_threshold(min_threshold)}"]
    if max_threshold is not None:
        parts.append(f"max-{format_threshold(max_threshold)}")
    return "_".join(parts)


def process_modes(
    dataset: str,
    modes: Sequence[str],
    raw_root: Path,
    output_root: Path,
    min_threshold: float,
    max_threshold: Optional[float],
) -> None:
    dataset_dir = raw_root / dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    dataset_output_dir = output_root / dataset
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    threshold_tag = build_threshold_tag(min_threshold, max_threshold)

    per_file_rows = []
    per_file_paths = []
    processed_files = 0
    total_samples = 0
    total_kept = 0

    for mode in modes:
        input_filename = f"{dataset}_train_{mode}_raw_fbl.jsonl"
        input_path = dataset_dir / input_filename
        if not input_path.is_file():
            print(
                f"[WARNING] Skipping missing file for mode='{mode}': {input_path}"
            )
            continue

        output_path = (
            dataset_output_dir
            / f"{input_path.stem}_threshold-{threshold_tag}.jsonl"
        )
        total, kept = filter_file(
            input_path, output_path, min_threshold, max_threshold
        )
        dropped = total - kept
        keep_rate = (kept / total * 100) if total else 0.0
        per_file_rows.append(
            [
                input_filename,
                total,
                kept,
                dropped,
                f"{keep_rate:.1f}%" if total else "n/a",
            ]
        )
        per_file_paths.append((input_filename, str(output_path.resolve())))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter dataset JSONL files by reasoning token floor and optional output token ceiling."
        )
    )
    parser.add_argument(
        "dataset",
        choices=DATASET_CHOICES,
        help="Dataset name to process.",
    )
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        action="append",
        dest="modes",
        help=(
            "Difficulty splits to process. Specify multiple times to filter a subset. "
            "Defaults to all splits when omitted."
        ),
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        required=True,
        help="Keep rows with reasoning_tokens >= this value.",
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        help="Keep rows with output_tokens <= this value (optional).",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Root directory that contains dataset folders under 1_fbl.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination directory to store filtered files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    modes: Sequence[str]
    modes = args.modes if args.modes else MODE_CHOICES

    process_modes(
        dataset=args.dataset,
        modes=modes,
        raw_root=args.raw_root,
        output_root=args.output_root,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
    )


if __name__ == "__main__":
    main()
