#!/usr/bin/env python3
"""Build step-5 splits: SFT keeps all filtered data, VAL is a stratified slice, GRPO backfills from alpaca."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

DATASET_CHOICES = ("gsm8k", "logiqa", "compmath")
MODE_NAMES = ("high", "medium", "low")
MODE_OUTPUT_ORDER = ("low", "medium", "high")
MODE_LABELS = {"low": "low", "medium": "medium", "high": "high"}

ALPACA_SOURCES = {
    "gsm8k": Path("data") / "gsm8k_train_alpaca.jsonl",
    "logiqa": Path("data") / "logiqa_train_alpaca.jsonl",
    "compmath": Path("data") / "competition_math_train_alpaca.jsonl",
}

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = BASE_DIR / "new_dataset" / "4_fbi"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "new_dataset" / "5_split"

GRPO_MODE_INSTRUCTIONS = {
    "low": "Respond concisely with minimal reasoning.\n",
    "medium": "Solve step-by-step.\n",
    "high": "Think deeply, verify, and self-correct.\n",
}


@dataclass
class Sample:
    sample_id: Optional[int]
    mode: str
    line: str
    data: Dict[str, Any]
    source: Path
    position: int
    reasoning_tokens: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send all filtered data to SFT, draw a length-diverse validation slice, "
            "and populate GRPO with examples missing from the alpaca source."
        )
    )
    parser.add_argument("--dataset", choices=DATASET_CHOICES, help="Dataset name.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing per-mode JSONL files (default: new_dataset/4_fbi).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination directory for the split outputs (default: new_dataset/5_split).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many files (useful for testing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report statistics without writing split files.",
    )
    parser.add_argument(
        "--ratio-sft",
        type=float,
        default=0.9,
        help="Target ratio for SFT split (default: 0.9).",
    )
    parser.add_argument(
        "--ratio-val",
        type=float,
        default=0.1,
        help="Target ratio for validation split (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when breaking ties in stratified sampling (default: 42).",
    )
    return parser.parse_args()


def normalize_id(raw_id: Any) -> Optional[int]:
    if raw_id is None:
        return None
    try:
        return int(str(raw_id).strip())
    except (ValueError, TypeError):
        return None


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


def detect_mode(path: Path) -> Optional[str]:
    name = path.stem.lower()
    for mode in MODE_NAMES:
        marker = f"_{mode}_"
        if marker in name:
            return mode
        if name.endswith(f"_{mode}"):
            return mode
        if name.startswith(f"{mode}_"):
            return mode
    return None


def iter_samples(path: Path, mode: str, start_index: int) -> Tuple[List[Sample], int]:
    samples: List[Sample] = []
    position = start_index
    with path.open("r", encoding="utf-8") as src:
        for raw_line in src:
            serialized = raw_line if raw_line.endswith("\n") else raw_line + "\n"
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            sample_id = normalize_id(payload.get("id"))
            tokens = read_numeric(payload.get("reasoning_tokens"))
            samples.append(
                Sample(
                    sample_id=sample_id,
                    mode=mode,
                    line=serialized,
                    data=payload,
                    source=path,
                    position=position,
                    reasoning_tokens=tokens,
                )
            )
            position += 1
    return samples, position


def load_samples(files: Sequence[Path], limit: Optional[int]) -> Tuple[List[Sample], Dict[str, int]]:
    samples: List[Sample] = []
    mode_counts = {mode: 0 for mode in MODE_NAMES}
    position = 0

    selected_files = sorted(files)
    if limit is not None:
        selected_files = selected_files[:limit]

    for path in selected_files:
        mode = detect_mode(path)
        if mode is None:
            print(f"[WARN] Could not infer mode for {path.name}; skipping.")
            continue
        mode_samples, position = iter_samples(path, mode, position)
        mode_counts[mode] += len(mode_samples)
        samples.extend(mode_samples)

    return samples, mode_counts


def reasoning_length(sample: Sample) -> float:
    if sample.reasoning_tokens is None or not math.isfinite(sample.reasoning_tokens):
        return float("inf")
    return float(sample.reasoning_tokens)


def shuffle_and_sort(samples: List[Sample], seed: int) -> List[Sample]:
    randomized = list(samples)
    random.Random(seed).shuffle(randomized)
    randomized.sort(key=lambda s: (reasoning_length(s), s.position))
    return randomized


def evenly_spaced_indices(total: int, amount: int) -> List[int]:
    if amount <= 0:
        return []
    if amount >= total:
        return list(range(total))
    selected: List[int] = []
    step = total / amount
    for k in range(amount):
        target = (k + 0.5) * step
        idx = max(0, min(total - 1, int(round(target)) - 1))
        while idx in selected and idx < total - 1:
            idx += 1
        while idx in selected and idx > 0:
            idx -= 1
        selected.append(idx)
    selected = sorted(set(selected))
    while len(selected) < amount:
        for idx in range(total):
            if idx not in selected:
                selected.append(idx)
                if len(selected) == amount:
                    break
    return sorted(selected)


def split_for_validation(
    samples: List[Sample], val_ratio: float, seed: int
) -> Tuple[List[Sample], List[Sample], int]:
    total_samples = len(samples)
    if total_samples == 0:
        return [], [], 0

    def sample_group_key(sample: Sample) -> str:
        if sample.sample_id is not None:
            return f"id:{sample.sample_id}"
        return f"pos:{sample.position}"

    grouped: Dict[str, List[Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample_group_key(sample), []).append(sample)
    groups = list(grouped.values())
    total_groups = len(groups)
    if total_groups == 0:
        return [], samples, 0

    val_target_samples = int(round(total_samples * val_ratio))
    val_group_count = int(round(total_groups * val_ratio))
    if val_group_count == 0 and val_ratio > 0:
        val_group_count = 1
    val_group_count = max(0, min(total_groups, val_group_count))
    if val_group_count == 0:
        return [], samples, val_target_samples

    def group_length(group: List[Sample]) -> float:
        return min(reasoning_length(sample) for sample in group)

    def group_position(group: List[Sample]) -> int:
        return min(sample.position for sample in group)

    randomized = list(groups)
    random.Random(seed).shuffle(randomized)
    randomized.sort(key=lambda group: (group_length(group), group_position(group)))
    chosen_indices = set(evenly_spaced_indices(len(randomized), val_group_count))

    val_groups = [group for idx, group in enumerate(randomized) if idx in chosen_indices]
    sft_groups = [group for idx, group in enumerate(randomized) if idx not in chosen_indices]
    val_samples = [sample for group in val_groups for sample in group]
    sft_samples = [sample for group in sft_groups for sample in group]
    return val_samples, sft_samples, val_target_samples


def group_by_mode(samples: List[Sample]) -> Dict[str, List[Sample]]:
    grouped: Dict[str, List[Sample]] = {mode: [] for mode in MODE_OUTPUT_ORDER}
    for sample in sorted(samples, key=lambda s: s.position):
        grouped.setdefault(sample.mode, []).append(sample)
    return grouped


def build_records(samples: List[Sample]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for sample in sorted(samples, key=lambda s: s.position):
        payload = sample.data or {}
        reasoning = payload.get("reasoning")
        prompt = payload.get("prompt")
        if not isinstance(reasoning, str) or not reasoning.strip():
            continue
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        record: Dict[str, Any] = {
            "id": payload.get("id"),
            "generation_index": payload.get("generation_index"),
            "prompt": prompt,
            "predict": payload.get("predict"),
            "label": payload.get("label"),
            "reasoning": reasoning,
            "reasoning_tokens": sample.reasoning_tokens,
        }
        records.append(record)
    return records


def write_records_by_mode(
    grouped: Dict[str, List[Sample]],
    destinations: Dict[str, Path],
    dry_run: bool,
) -> Dict[str, int]:
    written_counts: Dict[str, int] = {}
    for mode in MODE_OUTPUT_ORDER:
        records = build_records(grouped.get(mode, []))
        written_counts[mode] = len(records)
        target = destinations[mode]
        if dry_run:
            continue
        if not records:
            if target.exists():
                target.unlink()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as dst:
            for record in records:
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
    return written_counts


def gather_reasoning_values(grouped: Dict[str, List[Sample]]) -> Dict[str, List[float]]:
    values: Dict[str, List[float]] = {}
    for mode, samples in grouped.items():
        per_mode = [
            reasoning_length(sample)
            for sample in samples
            if sample.reasoning_tokens is not None and math.isfinite(sample.reasoning_tokens)
        ]
        values[mode] = per_mode
    return values


def build_histogram_edges(values: List[float], bins: int = 50) -> List[float]:
    if not values:
        return [0.0, 1.0]
    min_val = min(values)
    max_val = max(values)
    if math.isclose(min_val, max_val):
        max_val = min_val + 1.0
    step = (max_val - min_val) / bins
    edges = [min_val + i * step for i in range(bins)]
    edges.append(max_val)
    return edges


def histogram_counts(values: List[float], edges: List[float]) -> List[int]:
    if not values:
        return [0] * (len(edges) - 1)
    counts = [0] * (len(edges) - 1)
    min_edge = edges[0]
    max_edge = edges[-1]
    span = max_edge - min_edge
    if span <= 0:
        counts[0] = len(values)
        return counts
    for value in values:
        if value <= min_edge:
            index = 0
        elif value >= max_edge:
            index = len(counts) - 1
        else:
            ratio = (value - min_edge) / span
            index = min(int(ratio * len(counts)), len(counts) - 1)
        counts[index] += 1
    return counts


def save_histogram_overlay(
    dataset: str,
    reasoning_values: Dict[str, List[float]],
    edges: List[float],
    output_dir: Path,
) -> Optional[Path]:
    if plt is None:
        print("[WARN] matplotlib is not available; skipping histogram plots.")
        return None
    if not edges or len(edges) < 2:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    x_limits = (edges[0], edges[-1])
    colors = {"sft_train": "#3A7BD5", "val": "#2A9D8F"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    bin_centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    per_bucket_peaks: Dict[str, int] = {}

    for bucket_name, color in colors.items():
        values = reasoning_values.get(bucket_name, [])
        if not values:
            continue
        counts = histogram_counts(values, edges)
        per_bucket_peaks[bucket_name] = max(counts) if counts else 0
        ax.step(bin_centers, counts, where="mid", label=bucket_name, color=color, linewidth=2)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    max_peak = max(per_bucket_peaks.values()) if per_bucket_peaks else 0
    y_limits = (0, max_peak * 1.05 if max_peak else 1.0)

    ax.set_title(f"{dataset} reasoning_tokens distribution")
    ax.set_xlabel("reasoning_tokens")
    ax.set_ylabel("Frequency (count per bin)")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    path = output_dir / f"{dataset}_reasoning_hist_overlay.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_cdf_plot(
    dataset: str,
    reasoning_values: Dict[str, List[float]],
    x_limits: Tuple[float, float],
    output_dir: Path,
) -> Optional[Path]:
    if plt is None:
        return None
    xmin, xmax = x_limits
    if math.isclose(xmin, xmax):
        xmax = xmin + 1.0

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"sft_train": "#3A7BD5", "val": "#2A9D8F"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False

    for bucket_name, color in colors.items():
        values = reasoning_values.get(bucket_name, [])
        if not values:
            continue
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        ys = [(idx + 1) / n for idx in range(n)]
        ax.plot(sorted_vals, ys, label=bucket_name, color=color, linewidth=2)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_title(f"{dataset} reasoning_tokens CDF")
    ax.set_xlabel("reasoning_tokens")
    ax.set_ylabel("Cumulative probability")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    path = output_dir / f"{dataset}_reasoning_cdf.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def normalize_ratios(ratios: Dict[str, float]) -> Dict[str, float]:
    cleaned = {key: max(0.0, value) for key, value in ratios.items()}
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("At least one ratio value must be positive.")
    return {key: value / total for key, value in cleaned.items()}


def build_grpo_records_from_alpaca(
    dataset: str, seen_ids: Sequence[int]
) -> Tuple[List[Dict[str, Any]], Path]:
    source = ALPACA_SOURCES.get(dataset)
    if source is None:
        raise ValueError(f"No alpaca source registered for dataset '{dataset}'.")
    seen = set(seen_ids)
    records: List[Dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as src:
        for idx, raw_line in enumerate(src):
            if idx in seen:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            prompt = payload.get("prompt")
            response = payload.get("response")
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            if not isinstance(response, str) or not response.strip():
                continue
            response_clean = response.strip()
            prompt_clean = prompt.strip()
            base_record = {
                "id": idx,
                "prompt": prompt_clean,
                "predict": "",
                "label": response_clean,
            }
            for mode, instruction in GRPO_MODE_INSTRUCTIONS.items():
                mode_instruction = instruction.strip()
                record = dict(base_record)
                record["mode"] = mode
                if mode_instruction:
                    record["instruction"] = mode_instruction
                records.append(record)
    return records, source


def write_grpo_records(records: List[Dict[str, Any]], path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    if not records:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as dst:
        for record in records:
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_dataset(
    dataset: str,
    input_root: Path,
    output_root: Path,
    limit: Optional[int],
    dry_run: bool,
    ratios: Dict[str, float],
    seed: int,
) -> None:
    input_dir = input_root / dataset
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(input_dir.glob("*.jsonl"))
    if limit is not None:
        files = files[:limit]
    if not files:
        print(f"[{dataset}] No JSONL files detected under {input_dir}.")
        return

    samples, mode_counts = load_samples(files, limit=None)
    if not samples:
        print(f"[{dataset}] No valid samples collected from {len(files)} files.")
        return

    seen_ids = [sample.sample_id for sample in samples if sample.sample_id is not None]
    usable_samples = [
        sample
        for sample in samples
        if isinstance(sample.data.get("reasoning"), str)
        and sample.data.get("reasoning").strip()
        and isinstance(sample.data.get("prompt"), str)
        and sample.data.get("prompt").strip()
    ]

    normalized_ratios = normalize_ratios({"sft_train": ratios["sft_train"], "val": ratios["val"]})
    val_samples, sft_samples, val_target = split_for_validation(
        usable_samples, val_ratio=normalized_ratios["val"], seed=seed
    )

    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    outputs = {
        "sft_train": {mode: output_dir / f"{dataset}_sft_train_{mode}.jsonl" for mode in MODE_OUTPUT_ORDER},
        "val": {mode: output_dir / f"{dataset}_val_{mode}.jsonl" for mode in MODE_OUTPUT_ORDER},
    }

    grouped_sft = group_by_mode(sft_samples)
    grouped_val = group_by_mode(val_samples)

    print(f"[{dataset}] Mode counts (input): " + ", ".join(f"{mode}={count}" for mode, count in mode_counts.items()))
    print(
        f"[{dataset}] Selected {len(val_samples)} validation samples "
        f"({val_target} target, {len(usable_samples)} usable, {len(samples)} total)."
    )

    sft_written = write_records_by_mode(grouped_sft, outputs["sft_train"], dry_run=dry_run)
    val_written = write_records_by_mode(grouped_val, outputs["val"], dry_run=dry_run)

    per_bucket_rows: List[List[Any]] = []
    total_records = sum(sft_written.values()) + sum(val_written.values())
    for split_name, counts in (("sft_train", sft_written), ("val", val_written)):
        split_total = sum(counts.values())
        per_bucket_rows.append(
            [
                split_name,
                split_total,
                f"{(split_total / total_records * 100):.2f}%" if total_records else "0.00%",
                ", ".join(f"{mode}:{counts.get(mode,0)}" for mode in MODE_OUTPUT_ORDER),
            ]
        )
        dests = ", ".join(f"{mode}:{outputs[split_name][mode]}" for mode in MODE_OUTPUT_ORDER)
        print(
            f"  - {split_name}: samples={split_total} -> "
            f"{dests if not dry_run else '(dry run)'}"
        )

    if per_bucket_rows:
        headers = ["Split", "Samples", "Ratio", "Per-mode counts"]
        print(f"[{dataset}] Split summary:")
        print(tabulate(per_bucket_rows, headers=headers, tablefmt="grid"))

    reasoning_values = {
        "sft_train": [
            reasoning_length(sample)
            for sample in sft_samples
            if sample.reasoning_tokens is not None and math.isfinite(sample.reasoning_tokens)
        ],
        "val": [
            reasoning_length(sample)
            for sample in val_samples
            if sample.reasoning_tokens is not None and math.isfinite(sample.reasoning_tokens)
        ],
    }
    combined_values = [value for values in reasoning_values.values() for value in values]
    if combined_values:
        edges = build_histogram_edges(combined_values)
        plot_path = save_histogram_overlay(
            dataset=dataset, reasoning_values=reasoning_values, edges=edges, output_dir=plot_dir
        )
        cdf_path = save_cdf_plot(
            dataset=dataset,
            reasoning_values=reasoning_values,
            x_limits=(edges[0], edges[-1]),
            output_dir=plot_dir,
        )
        if plot_path:
            print(f"[{dataset}] Saved histogram overlay: {plot_path}")
        if cdf_path:
            print(f"[{dataset}] Saved CDF plot: {cdf_path}")
    else:
        print(f"[{dataset}] No reasoning_tokens values available; skipping histograms.")

    grpo_records, alpaca_source = build_grpo_records_from_alpaca(dataset, seen_ids=seen_ids)
    grpo_path = output_dir / f"{dataset}_grpo_train.jsonl"
    write_grpo_records(grpo_records, grpo_path, dry_run=dry_run)
    print(
        f"[{dataset}] GRPO backfill: {len(grpo_records)} missing ids from {alpaca_source} "
        f"-> {grpo_path if not dry_run else '(dry run)'}"
    )

    if dry_run:
        print("[INFO] Dry run mode; no files were written.")


def main() -> None:
    args = parse_args()
    process_dataset(
        dataset=args.dataset,
        input_root=args.input_root,
        output_root=args.output_root,
        limit=args.limit,
        dry_run=args.dry_run,
        ratios={"sft_train": args.ratio_sft, "val": args.ratio_val},
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
