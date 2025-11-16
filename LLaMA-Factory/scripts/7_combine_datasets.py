#!/usr/bin/env python3
"""Combine per-mode training JSONL files, reset indices, and summarize contributions."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from tabulate import tabulate
except ImportError as exc:  # pragma: no cover - tabulate required at runtime
    raise SystemExit(
        "Missing dependency 'tabulate'. Install it with `pip install tabulate`."
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

DATASETS = ("gsm8k", "logiqa", "compmath")
SPLITS = ("sft_train", "grpo_train", "val")
MODES = ("low", "medium", "high")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = BASE_DIR / "new_dataset" / "6_train"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "new_dataset" / "7_combined"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine per-mode training JSONL files across datasets, sort by reasoning length, "
            "reset indices, and produce summary stats/plots."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        help="Datasets to include (default: gsm8k logiqa compmath). Missing files are skipped.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing step-6 outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory to write combined files and plots.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing files.",
    )
    return parser.parse_args()


def extract_reasoning_text(response: object) -> Optional[str]:
    if not isinstance(response, str):
        return None
    start_tag = "<think>"
    end_tag = "</think>"
    start = response.find(start_tag)
    if start == -1:
        return None
    end = response.find(end_tag, start + len(start_tag))
    if end == -1:
        return None
    reasoning = response[start + len(start_tag) : end].strip()
    return reasoning or None


def read_training_file(path: Path, dataset: str, split: str) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    if not path.is_file():
        return entries
    with path.open("r", encoding="utf-8") as src:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record["dataset"] = dataset
            reasoning_tokens = record.get("reasoning_tokens")
            if isinstance(reasoning_tokens, (int, float)):
                record["reasoning_tokens"] = float(reasoning_tokens)
            else:
                response = record.get("response")
                if isinstance(response, str):
                    tokens = response.replace("<think>", "").replace("</think>", "").split()
                    record["reasoning_tokens"] = float(len(tokens))
                else:
                    record["reasoning_tokens"] = float("inf")
            if split == "grpo_train":
                reasoning_field = record.get("reasoning")
                if not isinstance(reasoning_field, str) or not reasoning_field.strip():
                    extracted = extract_reasoning_text(record.get("response"))
                    if extracted:
                        record["reasoning"] = extracted
            entries.append(record)
    return entries


def save_combined(entries: List[Dict[str, object]], path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    if not entries:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as dst:
        for entry in entries:
            dst.write(json.dumps(entry, ensure_ascii=False) + "\n")


def compute_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
    }


def process_combination(
    datasets: Sequence[str],
    split: str,
    mode: str,
    input_root: Path,
    output_root: Path,
    dry_run: bool,
) -> Tuple[int, Dict[str, int], Dict[str, Optional[float]], List[float]]:
    aggregated: List[Dict[str, object]] = []
    per_dataset_counts: Dict[str, int] = {}

    for dataset in datasets:
        input_path = input_root / dataset / f"{dataset}_{split}_{mode}_think.jsonl"
        entries = read_training_file(input_path, dataset=dataset, split=split)
        if not entries:
            continue
        per_dataset_counts[dataset] = len(entries)
        aggregated.extend(entries)

    if not aggregated:
        return 0, per_dataset_counts, compute_stats([]), []

    aggregated.sort(
        key=lambda record: (
            record.get("reasoning_tokens", float("inf")),
            record.get("dataset", ""),
        )
    )
    for index, record in enumerate(aggregated):
        record["combined_index"] = index

    combined_path = output_root / split / f"combined_{split}_{mode}.jsonl"
    save_combined(aggregated, combined_path, dry_run=dry_run)

    lengths = [float(record.get("reasoning_tokens", 0)) for record in aggregated]
    valid_lengths = [length for length in lengths if math.isfinite(length)]
    stats = compute_stats(valid_lengths)
    return len(aggregated), per_dataset_counts, stats, valid_lengths


def plot_sorted_lengths(lengths: List[float], output_dir: Path, title: str, filename: str) -> None:
    if plt is None or not lengths:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_lengths = sorted(lengths)
    x_values = list(range(1, len(sorted_lengths) + 1))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x_values, sorted_lengths, color="#3A7BD5", linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("Sample (sorted)")
    ax.set_ylabel("Reasoning length (tokens)")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def plot_contribution_pie(contributions: Dict[Tuple[str, str], int], output_dir: Path) -> None:
    if plt is None or not contributions:
        return
    labels = []
    sizes = []
    colors = []
    base_colors = {
        "gsm8k": ("#ccece6", "#66c2a4", "#006d2c"),
        "logiqa": ("#ffe0b2", "#ffb74d", "#e65100"),
        "compmath": ("#dedeff", "#9d94ff", "#5a4fcf"),
    }
    ordered_pairs = []
    for dataset in DATASETS:
        for mode in MODES:
            key = (dataset, mode)
            if key in contributions:
                ordered_pairs.append((dataset, mode))
    for key in contributions:
        if key not in ordered_pairs:
            ordered_pairs.append(key)

    for dataset, mode in ordered_pairs:
        count = contributions.get((dataset, mode), 0)
        if count <= 0:
            continue
        labels.append(f"{dataset}-{mode}")
        sizes.append(count)
        palette = base_colors.get(dataset)
        if palette:
            mode_index = ("low", "medium", "high").index(mode)
            color = palette[mode_index]
        else:
            color = "#999999"
        colors.append(color)
    if not sizes:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    total = sum(sizes)
    wedges, _, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct="%1.1f%%",
        startangle=0,
        pctdistance=0.75,
        textprops={"fontsize": 12},
    )
    ax.legend(
        wedges,
        labels,
        title="Dataset-Mode",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    centre_circle = plt.Circle((0, 0), 0.55, fc="white")
    fig.gca().add_artist(centre_circle)
    ax.text(
        0,
        0,
        f"{total:,}\nentries",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title("Mode contributions across datasets")
    fig.tight_layout()
    fig.savefig(output_dir / "combined_mode_contributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def process(args: argparse.Namespace) -> None:
    datasets = [dataset for dataset in args.datasets if dataset]
    if not datasets:
        raise ValueError("No datasets specified.")

    results: List[List[object]] = []
    stat_rows: List[List[object]] = []
    contributions: Dict[Tuple[str, str], int] = {}
    total_entries = 0
    all_lengths: List[float] = []

    for split in SPLITS:
        for mode in MODES:
            count, counts_per_dataset, stats, lengths = process_combination(
                datasets,
                split,
                mode,
                input_root=args.input_root,
                output_root=args.output_root,
                dry_run=args.dry_run,
            )
            total_entries += count
            all_lengths.extend(length for length in lengths if math.isfinite(length))

            row = [split, mode, count]
            for dataset in datasets:
                dataset_count = counts_per_dataset.get(dataset, 0)
                row.append(dataset_count)
                contributions[(dataset, mode)] = contributions.get((dataset, mode), 0) + dataset_count
            results.append(row)

            stat_rows.append(
                [
                    split,
                    mode,
                    count,
                    f"{stats['min']:.1f}" if stats.get("min") is not None else "n/a",
                    f"{stats['max']:.1f}" if stats.get("max") is not None else "n/a",
                    f"{stats['mean']:.1f}" if stats.get("mean") is not None else "n/a",
                    f"{stats['median']:.1f}" if stats.get("median") is not None else "n/a",
                ]
            )

    headers = ["Split", "Mode", "Combined"] + list(datasets)
    print("\n[Combined] Entry counts per split/mode:")
    print(tabulate(results, headers=headers, tablefmt="grid"))
    print(f"[Combined] Total entries processed: {total_entries}")

    stat_headers = ["Split", "Mode", "Count", "Min", "Max", "Mean", "Median"]
    print("\n[Combined] Reasoning length stats:")
    print(tabulate(stat_rows, headers=stat_headers, tablefmt="grid"))

    dataset_rows = []
    for dataset in datasets:
        total = sum(contributions.get((dataset, mode), 0) for mode in MODES)
        row = [dataset, total]
        for mode in MODES:
            row.append(contributions.get((dataset, mode), 0))
        dataset_rows.append(row)
    dataset_headers = ["Dataset", "Total"] + [f"{mode}_count" for mode in MODES]
    print("\n[Combined] Dataset contributions:")
    print(tabulate(dataset_rows, headers=dataset_headers, tablefmt="grid"))

    if not args.dry_run:
        plot_dir = args.output_root / "plots"
        plot_sorted_lengths(
            lengths=all_lengths,
            output_dir=plot_dir,
            title="Combined reasoning lengths (all splits/modes)",
            filename="combined_all_lengths.png",
        )
        plot_contribution_pie(contributions, plot_dir)
    else:
        print("[INFO] Dry run mode; no files were written.")


def main() -> None:
    args = parse_args()
    process(args)


if __name__ == "__main__":
    main()
