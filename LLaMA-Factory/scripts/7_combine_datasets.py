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
    from matplotlib.patches import Wedge
except ImportError:  # pragma: no cover - optional dependency
    plt = None

DATASETS = ("gsm8k", "logiqa", "compmath")
SPLITS = ("sft_train", "grpo_train", "val")
SPLIT_MODES = {
    "sft_train": ("low", "medium", "high"),
    "val": ("low", "medium", "high"),
    "grpo_train": ("grpo",),
}
ALL_MODES = ("low", "medium", "high", "grpo")

GRPO_MODE_INSTRUCTIONS = {
    "low": "Respond concisely with minimal reasoning.\n",
    "medium": "Solve step-by-step.\n",
    "high": "Think deeply, verify, and self-correct.\n",
}
GRPO_INSTRUCTION_PROMPT = ""  # kept for compatibility but intentionally empty
PROMPT_SOURCE_KEYS = ("question", "input")
PROMPT_CLEANUP_PHRASES = (
    " Please reason step by step, and put your final answer within \\boxed{}. Do NOT include explanations or reasoning in the final answer - only the numeric value in \\boxed{}",
    " Please reason step by step, and put your final answer as a single letter (A/B/C/D) within \\boxed{}. Do NOT include explanations or reasoning in the final answer - only the letter in \\boxed{}.",
    " Please reason step by step, and put your final answer within \\boxed{}. Do NOT include explanations or reasoning in the final answer - only the mathematical expression or value in \\boxed{}.",
)

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
    parser.add_argument(
        "--sort-order",
        choices=("asc", "desc"),
        default="asc",
        help="Sort combined data by reasoning length ascending (asc) or descending (desc).",
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


def strip_mode_instruction_prefix(text: str) -> str:
    stripped = text
    for instruction in GRPO_MODE_INSTRUCTIONS.values():
        marker = instruction.strip()
        if marker and stripped.startswith(marker):
            stripped = stripped[len(marker) :].lstrip()
            break
    return stripped


def strip_general_instruction_suffix(text: str) -> str:
    if not GRPO_INSTRUCTION_PROMPT:
        return text
    stripped = text.rstrip()
    marker = GRPO_INSTRUCTION_PROMPT.strip()
    if marker and stripped.endswith(marker):
        stripped = stripped[: -len(marker)].rstrip()
    return stripped


def normalize_prompt_source_from_prompt(prompt_value: str) -> str:
    text = strip_general_instruction_suffix(prompt_value)
    text = strip_mode_instruction_prefix(text)
    return text.strip()


def extract_prompt_source(record: Dict[str, object]) -> Optional[str]:
    for source_key in PROMPT_SOURCE_KEYS:
        candidate = record.get(source_key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    prompt_candidate = record.get("prompt")
    if isinstance(prompt_candidate, str) and prompt_candidate.strip():
        normalized = normalize_prompt_source_from_prompt(prompt_candidate)
        if normalized:
            return normalized
    return None


def format_question_prompt(base_text: str) -> str:
    stripped = strip_prompt_cleanup(base_text.strip())
    if stripped.lower().startswith("question:") or stripped.lower().count("question:")==1:
        return stripped
    return f"Question: {stripped}"


def strip_prompt_cleanup(text: str) -> str:
    cleaned = text
    for phrase in PROMPT_CLEANUP_PHRASES:
        if phrase and phrase in cleaned:
            cleaned = cleaned.replace(phrase, "")
    return cleaned.strip()


def build_prompt_with_instruction(record: Dict[str, object]) -> Optional[str]:
    prompt_source = extract_prompt_source(record)
    if not prompt_source:
        return None
    question_text = format_question_prompt(prompt_source)
    return question_text


def build_instruction_text(mode: str) -> Optional[str]:
    mode_instruction = GRPO_MODE_INSTRUCTIONS.get(mode, "").strip()
    if not mode_instruction:
        return None
    return mode_instruction


def ensure_instruction_field(record: Dict[str, object], mode: str) -> None:
    instruction_text = build_instruction_text(mode)
    if not instruction_text:
        return
    existing = record.get("instruction")
    if isinstance(existing, str) and existing.strip() == instruction_text:
        return
    record["instruction"] = instruction_text


def read_training_file(
    path: Path, dataset: str, split: str, mode: str
) -> List[Dict[str, object]]:
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
                length_value = float(reasoning_tokens)
            else:
                response = record.get("response")
                if isinstance(response, str):
                    tokens = response.replace("<think>", "").replace("</think>", "").split()
                    length_value = float(len(tokens))
                else:
                    length_value = float("inf")
            if split != "grpo_train":
                record["reasoning_tokens"] = length_value
            else:
                record.pop("reasoning_tokens", None)
            record["_reasoning_length"] = length_value
            record_mode = mode
            record_mode_raw = record.get("mode")
            if isinstance(record_mode_raw, str) and record_mode_raw.strip():
                record_mode = record_mode_raw.strip().lower()
            if record_mode not in ALL_MODES:
                record_mode = mode
            record["mode"] = record_mode
            if split == "grpo_train":
                prompt_value = record.get("prompt")
                if not isinstance(prompt_value, str) or not prompt_value.strip():
                    prompt_text = build_prompt_with_instruction(record)
                    if prompt_text:
                        record["prompt"] = prompt_text
                else:
                    record["prompt"] = prompt_value.strip()
                if not isinstance(record.get("instruction"), str) or not record["instruction"].strip():
                    ensure_instruction_field(record, record_mode)
            elif split == "sft_train":
                prompt_text = build_prompt_with_instruction(record)
                if prompt_text:
                    record["prompt"] = prompt_text
                ensure_instruction_field(record, record_mode)
            elif split == "val":
                prompt_text = build_prompt_with_instruction(record)
                if prompt_text:
                    record["prompt"] = prompt_text
                ensure_instruction_field(record, record_mode)
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
            serialized = {key: value for key, value in entry.items() if not str(key).startswith("_")}
            dst.write(json.dumps(serialized, ensure_ascii=False) + "\n")


def compute_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
    }


def build_sort_key(descending: bool):
    def sort_key(record: Dict[str, object]) -> Tuple[float, str]:
        length = float(record.get("_reasoning_length", float("inf")))
        dataset = record.get("dataset", "")
        if math.isfinite(length):
            adjusted_length = -length if descending else length
        else:
            adjusted_length = float("inf")
        return (adjusted_length, dataset)

    return sort_key


def process_combination(
    datasets: Sequence[str],
    split: str,
    mode: str,
    input_root: Path,
    output_root: Path,
    dry_run: bool,
    sort_descending: bool,
) -> Tuple[int, Dict[str, int], Dict[str, Optional[float]], List[float]]:
    aggregated: List[Dict[str, object]] = []
    per_dataset_counts: Dict[str, int] = {}

    for dataset in datasets:
        if split == "grpo_train" and mode == "grpo":
            input_path = input_root / dataset / f"{dataset}_{split}_think.jsonl"
        else:
            input_path = input_root / dataset / f"{dataset}_{split}_{mode}_think.jsonl"
        entries = read_training_file(input_path, dataset=dataset, split=split, mode=mode)
        if not entries:
            continue
        per_dataset_counts[dataset] = len(entries)
        aggregated.extend(entries)

    if not aggregated:
        return 0, per_dataset_counts, compute_stats([]), []

    if split != "grpo_train":
        aggregated.sort(key=build_sort_key(sort_descending))
    for index, record in enumerate(aggregated):
        record["combined_index"] = index

    if split == "grpo_train" and mode == "grpo":
        combined_name = f"combined_{split}.jsonl"
    else:
        combined_name = f"combined_{split}_{mode}.jsonl"
    combined_path = output_root / split / combined_name
    save_combined(aggregated, combined_path, dry_run=dry_run)

    lengths = [float(record.get("_reasoning_length", float("inf"))) for record in aggregated]
    valid_lengths = [length for length in lengths if math.isfinite(length)]
    stats = compute_stats(valid_lengths)
    return len(aggregated), per_dataset_counts, stats, valid_lengths


def plot_sorted_lengths(
    lengths: List[float],
    output_dir: Path,
    title: str,
    filename: str,
    descending: bool,
) -> None:
    if plt is None or not lengths:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_lengths = sorted(lengths, reverse=descending)
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


def plot_contribution_pie(
    sft_train_counts: Dict[Tuple[str, str], int],
    val_counts: Dict[Tuple[str, str], int],
    grpo_counts: Dict[Tuple[str, str], int],
    output_dir: Path,
    datasets: Sequence[str],
    mode_order: Sequence[str],
) -> None:
    if plt is None:
        return

    base_colors = {
        "gsm8k": ("#ccece6", "#66c2a4", "#006d2c"),
        "logiqa": ("#ffe0b2", "#ffb74d", "#e65100"),
        "compmath": ("#dedeff", "#9d94ff", "#5a4fcf"),
    }
    outer_labels: List[str] = []
    outer_sizes: List[int] = []
    outer_colors: List[str] = []
    outer_val_sizes: List[int] = []
    inner_labels: List[str] = []
    inner_sizes: List[int] = []
    inner_colors: List[str] = []

    outer_modes = [mode for mode in mode_order if mode in ("low", "medium", "high")]
    for dataset in datasets:
        palette = base_colors.get(dataset)
        for mode in outer_modes:
            train_count = sft_train_counts.get((dataset, mode), 0)
            val_count = val_counts.get((dataset, mode), 0)
            total = train_count + val_count
            if total <= 0:
                continue
            outer_labels.append(f"{dataset}-{mode}")
            outer_sizes.append(total)
            outer_val_sizes.append(val_count)
            if palette:
                mode_index = ("low", "medium", "high").index(mode)
                outer_colors.append(palette[mode_index])
            else:
                outer_colors.append("#999999")

        grpo_count = grpo_counts.get((dataset, "grpo"), 0)
        if grpo_count > 0:
            inner_labels.append(f"{dataset}-grpo")
            inner_sizes.append(grpo_count)
            if palette:
                inner_colors.append(palette[-1])
            else:
                inner_colors.append("#b0b0b0")

    if not outer_sizes and not inner_sizes:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))

    def autopct_fmt(values: List[int]):
        total = sum(values)

        def _fmt(pct: float) -> str:
            if total <= 0 or pct < 1.0:
                return ""
            value = pct * total / 100.0
            if value < 1:
                return ""
            return f"{pct:.1f}%"

        return _fmt

    if outer_sizes:
        ax.pie(
            outer_sizes,
            radius=1.0,
            labels=None,
            colors=outer_colors,
            autopct=autopct_fmt(outer_sizes),
            pctdistance=0.88,
            startangle=0,
            textprops={"fontsize": 11},
            wedgeprops=dict(width=0.3, edgecolor="white"),
        )
    if outer_sizes and any(val > 0 for val in outer_val_sizes):
        val_wedges = ax.pie(
            outer_val_sizes,
            radius=0.82,
            labels=None,
            colors=["#555555"] * len(outer_val_sizes),
            startangle=0,
            textprops={"fontsize": 9, "color": "white"},
            wedgeprops=dict(width=0.12, edgecolor="white"),
        )[0]
    else:
        val_wedges = []

    if inner_sizes:
        inner_wedges, _, _ = ax.pie(
            inner_sizes,
            radius=0.7,
            labels=None,
            colors=inner_colors,
            autopct=autopct_fmt(inner_sizes),
            pctdistance=0.65,
            startangle=0,
            textprops={"fontsize": 10},
            wedgeprops=dict(width=0.3, edgecolor="white"),
        )
    else:
        inner_wedges = []

    for wedge, total, val in zip(val_wedges, outer_sizes, outer_val_sizes):
        if total <= 0 or val <= 0:
            continue
        angle = math.radians((wedge.theta2 + wedge.theta1) / 2)
        radius = 0.76
        ratio = val / total * 100
        ax.text(
            radius * math.cos(angle),
            radius * math.sin(angle),
            f"{ratio:.1f}%",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
        )

    outer_total = sum(outer_sizes)
    inner_total = sum(inner_sizes)
    center_lines = []
    center_lines.append(f"SFT\n{outer_total:,}" if outer_total else "SFT\n0")
    center_lines.append(f"GRPO\n{inner_total:,}" if inner_total else "GRPO\n0")
    ax.text(
        0,
        0.15,
        center_lines[0],
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0,
        -0.1,
        center_lines[1],
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    if outer_labels:
        sft_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10)
            for color in outer_colors
        ]
        ax.legend(
            sft_handles,
            [
                f"{label}: {size} (VAL {((val / size) * 100 if size else 0):.1f}%)"
                for label, size, val in zip(outer_labels, outer_sizes, outer_val_sizes)
            ],
            title="SFT (train+val)",
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
    if inner_labels:
        grpo_handles = [
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color, markersize=10)
            for color in inner_colors
        ]
        ax.add_artist(
            plt.legend(
                grpo_handles,
                [f"{label}: {size}" for label, size in zip(inner_labels, inner_sizes)],
                title="GRPO",
                loc="lower left",
                bbox_to_anchor=(1, 0),
            )
        )

    ax.set_title("SFT vs GRPO contributions\n(VAL ratio per dataset-mode)")
    fig.tight_layout()
    fig.savefig(output_dir / "combined_mode_contributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def process(args: argparse.Namespace) -> None:
    datasets = [dataset for dataset in args.datasets if dataset]
    if not datasets:
        raise ValueError("No datasets specified.")
    sort_descending = args.sort_order == "desc"

    results: List[List[object]] = []
    stat_rows: List[List[object]] = []
    contributions: Dict[Tuple[str, str], int] = {}
    sft_train_mode_counts: Dict[Tuple[str, str], int] = {}
    val_mode_counts: Dict[Tuple[str, str], int] = {}
    grpo_mode_counts: Dict[Tuple[str, str], int] = {}
    total_entries = 0
    all_lengths: List[float] = []

    for split in SPLITS:
        for mode in SPLIT_MODES[split]:
            count, counts_per_dataset, stats, lengths = process_combination(
                datasets,
                split,
                mode,
                input_root=args.input_root,
                output_root=args.output_root,
                dry_run=args.dry_run,
                sort_descending=sort_descending,
            )
            total_entries += count
            all_lengths.extend(length for length in lengths if math.isfinite(length))

            row = [split, mode, count]
            for dataset in datasets:
                dataset_count = counts_per_dataset.get(dataset, 0)
                row.append(dataset_count)
                contributions[(dataset, mode)] = contributions.get((dataset, mode), 0) + dataset_count
                key = (dataset, mode)
                if split == "sft_train" and dataset_count > 0:
                    sft_train_mode_counts[key] = dataset_count
                elif split == "val" and dataset_count > 0:
                    val_mode_counts[key] = dataset_count
                elif split == "grpo_train" and dataset_count > 0:
                    grpo_mode_counts[key] = dataset_count
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
    mode_order = [mode for mode in ALL_MODES if any(contributions.get((d, mode), 0) for d in datasets)]
    for dataset in datasets:
        total = sum(contributions.get((dataset, mode), 0) for mode in mode_order)
        row = [dataset, total]
        for mode in mode_order:
            row.append(contributions.get((dataset, mode), 0))
        dataset_rows.append(row)
    dataset_headers = ["Dataset", "Total"] + [f"{mode}_count" for mode in mode_order]
    print("\n[Combined] Dataset contributions:")
    print(tabulate(dataset_rows, headers=dataset_headers, tablefmt="grid"))

    if not args.dry_run:
        plot_dir = args.output_root / "plots"
        plot_sorted_lengths(
            lengths=all_lengths,
            output_dir=plot_dir,
            title="Combined reasoning lengths (all splits/modes)",
            filename="combined_all_lengths.png",
            descending=sort_descending,
        )
        plot_contribution_pie(
            sft_train_counts=sft_train_mode_counts,
            val_counts=val_mode_counts,
            grpo_counts=grpo_mode_counts,
            output_dir=plot_dir,
            datasets=datasets,
            mode_order=mode_order if mode_order else ALL_MODES,
        )
    else:
        print("[INFO] Dry run mode; no files were written.")


def main() -> None:
    args = parse_args()
    process(args)


if __name__ == "__main__":
    main()
