#!/usr/bin/env python3
"""Evaluate JSONL inference outputs by matching boxed answers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tabulate import tabulate
except ImportError as exc:  # pragma: no cover - tabulate required at runtime
    raise SystemExit(
        "Missing dependency 'tabulate'. Install it with `pip install tabulate`."
    ) from exc

BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or more JSONL inference files by comparing the boxed answers "
            "in 'predict' against those in 'label'. Multiple files can be provided as a "
            "comma-separated list."
        )
    )
    parser.add_argument(
        "files",
        help="Comma-separated list of JSONL files to evaluate.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require answers to match exactly (default trims whitespace and normalizes spaces).",
    )
    parser.add_argument(
        "--wrong-dir",
        type=Path,
        default=Path("scripts/sft/9_eval"),
        help="Directory to save wrong predictions per input file (default: 9_eval).",
    )
    return parser.parse_args()


def extract_boxed(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str):
        return None
    matches = BOXED_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def normalize(value: Optional[str], strict: bool) -> Optional[str]:
    if value is None:
        return None
    if strict:
        return value
    return " ".join(value.split()).lower()


def evaluate_file(path: Path, strict: bool) -> Tuple[int, int, List[Dict[str, object]]]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    total = 0
    correct = 0
    wrong_examples: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as src:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            raw_predict = extract_boxed(record.get("predict"))
            raw_label = extract_boxed(record.get("label"))
            predicted = normalize(raw_predict, strict)
            label = normalize(raw_label, strict)
            if label is None or predicted is None:
                continue
            total += 1
            if predicted == label:
                correct += 1
            else:
                wrong_examples.append(
                    {
                        "file": str(path),
                        "predict_raw": raw_predict,
                        "label_raw": raw_label,
                        "predict_normalized": predicted,
                        "label_normalized": label,
                        "record": record,
                    }
                )
    return total, correct, wrong_examples


def format_stats(total: int, correct: int) -> Tuple[int, int, str]:
    accuracy = (correct / total * 100) if total else 0.0
    return total, correct, f"{accuracy:.2f}%"


def save_wrong_examples(path: Path, examples: List[Dict[str, object]]) -> None:
    if not examples:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as dst:
        for example in examples:
            dst.write(json.dumps(example, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    files = [Path(item.strip()) for item in args.files.split(",") if item.strip()]
    if not files:
        raise ValueError("No valid files provided.")

    rows: List[List[object]] = []
    grand_total = 0
    grand_correct = 0

    wrong_dir = args.wrong_dir
    for file_path in files:
        total, correct, wrong_examples = evaluate_file(file_path, strict=args.strict)
        rows.append([str(file_path), total, correct, format_stats(total, correct)[2]])
        grand_total += total
        grand_correct += correct
        if wrong_dir:
            wrong_path = wrong_dir / f"{file_path.stem}_wrong.jsonl"
            save_wrong_examples(wrong_path, wrong_examples)

    headers = ["File", "Evaluated Samples", "Correct", "Accuracy"]
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    total_stats = format_stats(grand_total, grand_correct)
    print(
        f"\n[Summary] total={grand_total}, correct={grand_correct}, "
        f"accuracy={total_stats[2]}"
    )


if __name__ == "__main__":
    main()
