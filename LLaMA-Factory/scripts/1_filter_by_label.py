#!/usr/bin/env python3
"""Filter JSONL samples by matching prediction and label answers."""

import argparse
import json
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

try:
    from tabulate import tabulate
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'tabulate'. Install it with `pip install tabulate`."
    ) from exc

DATASET_CHOICES = ("gsm8k", "logiqa", "compmath")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = BASE_DIR / "new_dataset" / "0_raw"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "new_dataset" / "1_fbl"
DEFAULT_REJECT_ROOT = DEFAULT_OUTPUT_ROOT / "rejected"
DEFAULT_OUTPUT_SUFFIX = "_fbl"
DEFAULT_REJECT_SUFFIX = "_rejected"

BOXED_PATTERN = re.compile(r"\\boxed\{([^}]+)\}")
HASHED_PATTERN = re.compile(r"####\s*([^\n]+)")


def _is_prediction_complete(pred: Optional[str]) -> bool:
    if not isinstance(pred, str):
        return False

    stripped = pred.strip()
    if not stripped:
        return False

    matches = BOXED_PATTERN.findall(stripped)
    return len(matches) > 0 and any(m.strip() for m in matches)


def extract_answer(value: Any) -> Optional[str]:
    """Return the first boxed answer or the text after '####' marker."""
    if not isinstance(value, str):
        return None

    stripped = value.strip()
    if not stripped:
        return None

    matches = BOXED_PATTERN.findall(stripped)
    for match in matches:
        candidate = match.strip()
        if candidate:
            return candidate

    hash_match = HASHED_PATTERN.search(stripped)
    if hash_match:
        candidate = hash_match.group(1).strip()
        if candidate:
            return candidate
    return None


def extract_label_answer(dataset: str, label_raw: Any) -> Optional[str]:
    """Extract label answer using dataset-specific rules."""
    if dataset in {"gsm8k", "compmath"}:
        return extract_answer(label_raw).replace("<|end|><|return|>", "")
    if dataset == "logiqa":
        if not isinstance(label_raw, str):
            return None
        cleaned = label_raw.replace("<|end|><|return|>", "").replace("<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>", "").strip()
        return cleaned or None
    return None


def filter_samples(
    dataset: str,
    input_path: Path,
    output_path: Path,
    rejected_path: Optional[Path],
) -> tuple[int, int]:
    """Write samples whose prediction answer matches the label answer."""
    kept = 0
    rejected = 0
    with open(input_path, "r", encoding="utf-8") as src, open(
        output_path, "w", encoding="utf-8"
    ) as dst, (
        open(rejected_path, "w", encoding="utf-8")
        if rejected_path
        else nullcontext()
    ) as rejected_file:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                if rejected_file is not None:
                    rejected_file.write(
                        raw_line if raw_line.endswith("\n") else raw_line + "\n"
                    )
                    rejected += 1
                continue

            predict_raw = sample.get("predict")
            label_raw = sample.get("label")
            if not _is_prediction_complete(predict_raw):
                if rejected_file is not None:
                    rejected_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    rejected += 1
                continue

            predict_answer = extract_answer(predict_raw)
            label_answer = extract_label_answer(dataset, label_raw)

            if (
                predict_answer is not None
                and label_answer is not None
                and predict_answer == label_answer
            ):
                dst.write(json.dumps(sample, ensure_ascii=False) + "\n")
                kept += 1
            elif rejected_file is not None:
                rejected_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                rejected += 1
    return kept, rejected


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for path if it does not already exist."""
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def derive_target_path(source: Path, target_dir: Optional[Path], suffix: str) -> Path:
    """Build output path using suffix and optional target directory."""
    dest_dir = target_dir if target_dir is not None else source.parent
    filename = f"{source.stem}{suffix}{source.suffix or '.jsonl'}"
    return dest_dir / filename


def process_dataset(
    dataset: str,
    input_root: Path,
    output_root: Path,
    reject_root: Optional[Path],
) -> None:
    input_dir = input_root / dataset
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = output_root / dataset
    reject_dir = reject_root / dataset if reject_root is not None else None

    input_files = sorted(input_dir.glob("*.jsonl"))
    if not input_files:
        print(f"[INFO] No JSONL files found in {input_dir}; nothing to do.")
        return

    processed_files = 0
    total_kept = 0
    total_rejected = 0
    per_file_rows = []
    per_file_paths = []

    for input_path in input_files:
        output_path = derive_target_path(input_path, output_dir, DEFAULT_OUTPUT_SUFFIX)
        ensure_parent_dir(output_path)
        rejected_path = None
        if reject_dir is not None:
            rejected_path = derive_target_path(
                input_path, reject_dir, DEFAULT_REJECT_SUFFIX
            )
            ensure_parent_dir(rejected_path)
        kept, rejected = filter_samples(dataset, input_path, output_path, rejected_path)
        total = kept + rejected
        per_file_rows.append(
            [input_path.name, total, kept, rejected]
        )
        per_file_paths.append(
            (
                input_path.name,
                str(output_path),
                str(rejected_path) if rejected_path is not None else None,
            )
        )
        processed_files += 1
        total_kept += kept
        total_rejected += rejected

    total_samples = total_kept + total_rejected
    headers = ["File", "Total", "Kept", "Rejected"]
    print(f"[{dataset}] Per-file stats:")
    print(tabulate(per_file_rows, headers=headers, tablefmt="grid"))
    print(f"[{dataset}] Output paths:")
    for name, filtered, rejected in per_file_paths:
        print(f"  {name}: filtered -> {filtered}")
        if rejected is not None:
            print(f"           rejected -> {rejected}")
    print(
        f"[{dataset}] Summary: processed {processed_files} files, "
        f"kept {total_kept} samples, rejected {total_rejected} samples, "
        f"total {total_samples} samples."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter dataset samples where prediction matches label."
    )
    parser.add_argument("dataset", choices=DATASET_CHOICES, help="Dataset name.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing dataset sub-folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Base directory for filtered outputs.",
    )
    parser.add_argument(
        "--rejected-root",
        type=Path,
        default=DEFAULT_REJECT_ROOT,
        help="Base directory for rejected samples.",
    )
    parser.add_argument(
        "--skip-rejected",
        action="store_true",
        help="Skip writing rejected samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_root = args.input_root
    output_root = args.output_root
    rejected_root = None if args.skip_rejected else args.rejected_root

    output_root.mkdir(parents=True, exist_ok=True)
    if rejected_root is not None:
        rejected_root.mkdir(parents=True, exist_ok=True)

    process_dataset(args.dataset, input_root, output_root, rejected_root)
    print(f"[{args.dataset}] Done filtering.")


if __name__ == "__main__":
    main()
