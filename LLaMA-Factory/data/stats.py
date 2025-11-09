#!/usr/bin/env python3
"""Compute reasoning token statistics for SFT datasets."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_FILES = [
    "combined_sft_train_high.jsonl",
    "combined_sft_train_low.jsonl",
    "combined_sft_train_medium.jsonl",
]


@dataclass(frozen=True)
class Stats:
    count: int
    missing: int
    total: float
    mean: float
    median: float
    p25: float
    p75: float
    p05: float
    p95: float
    minimum: float
    maximum: float


def load_reasoning_tokens(path: Path) -> tuple[List[float], int]:
    """Return all reasoning_tokens values plus missing count for the given path."""
    values: List[float] = []
    missing = 0
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            value = payload.get("reasoning_tokens")
            if value is None:
                missing += 1
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                raise ValueError(f"Non-numeric reasoning_tokens in {path}") from None
    if missing:
        print(f"[WARN] {path.name}: {missing} samples missing reasoning_tokens")
    return values, missing


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Return the q-th percentile (0-100) using linear interpolation."""
    if not 0 <= q <= 100:
        raise ValueError("Percentile q must be in [0, 100]")
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = q / 100 * (len(sorted_values) - 1)
    lower_idx = math.floor(rank)
    upper_idx = math.ceil(rank)
    lower = sorted_values[lower_idx]
    upper = sorted_values[upper_idx]
    if lower_idx == upper_idx:
        return lower
    weight = rank - lower_idx
    return lower + weight * (upper - lower)


def compute_stats(values: Sequence[float], missing: int = 0) -> Stats:
    if not values:
        return Stats(0, missing, 0.0, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)
    sorted_vals = sorted(values)
    total = float(sum(sorted_vals))
    count = len(sorted_vals)
    mean = total / count
    return Stats(
        count=count,
        missing=missing,
        total=total,
        mean=mean,
        median=percentile(sorted_vals, 50),
        p25=percentile(sorted_vals, 25),
        p75=percentile(sorted_vals, 75),
        p05=percentile(sorted_vals, 5),
        p95=percentile(sorted_vals, 95),
        minimum=sorted_vals[0],
        maximum=sorted_vals[-1],
    )


def format_stats(label: str, stats: Stats) -> str:
    if stats.count == 0:
        return f"{label:<30} | count=0 | missing={stats.missing}"
    return (
        f"{label:<30} | count={stats.count:6d} | mean={stats.mean:8.2f} | "
        f"median={stats.median:8.2f} | min={stats.minimum:6.0f} | max={stats.maximum:7.0f} | "
        f"P05={stats.p05:7.2f} | P25={stats.p25:7.2f} | P75={stats.p75:7.2f} | P95={stats.p95:7.2f} | "
        f"missing={stats.missing}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        help="Relative or absolute paths to JSONL datasets (default: combined SFT splits).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_args = args.files or DEFAULT_FILES
    all_values: List[float] = []
    combined_missing = 0

    for file_arg in file_args:
        path = Path(file_arg)
        if not path.is_absolute():
            path = DATA_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        values, missing = load_reasoning_tokens(path)
        combined_missing += missing
        all_values.extend(values)
        stats = compute_stats(values, missing)
        print(format_stats(path.name, stats))

    print("-" * 120)
    combined_stats = compute_stats(all_values, combined_missing)
    print(format_stats("combined", combined_stats))


if __name__ == "__main__":
    main()
