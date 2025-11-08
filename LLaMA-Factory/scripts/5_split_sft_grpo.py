#!/usr/bin/env python3
"""Split filtered JSONL data into SFT/GRPO train sets and a shared validation set."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
MODE_LABELS = {
    "low": "low",
    "medium": "medium",
    "high": "high",
}
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = BASE_DIR / "new_dataset" / "4_fbi"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "new_dataset" / "5_split"


@dataclass
class Sample:
    sample_id: str
    mode: str
    line: str
    data: Dict[str, Any]
    source: Path
    position: int
    reasoning_tokens: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build stage-5 splits where ids appearing in all three modes (high/medium/low) "
            "feed SFT train, ids in exactly two modes feed GRPO train, and ids in only one "
            "mode are reserved for the shared validation set."
        )
    )
    parser.add_argument("dataset", choices=DATASET_CHOICES, help="Dataset name.")
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
        default=0.4,
        help="Target ratio for SFT split (default: 0.5).",
    )
    parser.add_argument(
        "--ratio-grpo",
        type=float,
        default=0.4,
        help="Target ratio for GRPO split (default: 0.45).",
    )
    parser.add_argument(
        "--ratio-val",
        type=float,
        default=0.1,
        help="Target ratio for validation split (default: 0.05).",
    )
    return parser.parse_args()


def normalize_id(raw_id: Any) -> Optional[str]:
    if raw_id is None:
        return None
    text = str(raw_id).strip()
    return text or None


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
            if not sample_id:
                continue

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


def collect_samples(files: List[Path]) -> Tuple[Dict[str, List[Sample]], Dict[str, int]]:
    samples_by_id: Dict[str, List[Sample]] = {}
    mode_counts = {mode: 0 for mode in MODE_NAMES}
    position = 0

    for path in files:
        mode = detect_mode(path)
        if mode is None:
            print(f"[WARN] Could not infer mode for {path.name}; skipping.")
            continue
        mode_samples, position = iter_samples(path, mode, position)
        mode_counts[mode] += len(mode_samples)
        for sample in mode_samples:
            samples_by_id.setdefault(sample.sample_id, []).append(sample)

    return samples_by_id, mode_counts


def categorize_samples(samples_by_id: Dict[str, List[Sample]]) -> Dict[str, Dict[str, List[Sample]]]:
    buckets: Dict[str, Dict[str, List[Sample]]] = {
        "sft_train": {},
        "grpo_train": {},
        "val": {},
    }
    for sample_id, entries in samples_by_id.items():
        modes = {sample.mode for sample in entries}
        if len(modes) >= 3:
            target = "sft_train"
        elif len(modes) == 2:
            target = "grpo_train"
        else:
            target = "val"
        buckets[target][sample_id] = sorted(entries, key=lambda sample: sample.position)
    return buckets


def group_samples_by_id(entries: List[Sample]) -> Dict[str, List[Sample]]:
    grouped: Dict[str, List[Sample]] = {}
    for sample in entries:
        grouped.setdefault(sample.sample_id, []).append(sample)
    for sample_id in grouped:
        grouped[sample_id].sort(key=lambda item: item.position)
    return grouped


def _group_reasoning_metric(samples: List[Sample]) -> float:
    values = [sample.reasoning_tokens for sample in samples if sample.reasoning_tokens is not None]
    if not values:
        return float("inf")
    return statistics.fmean(values)


def _group_order_key(samples: List[Sample], sort_by_reasoning: bool) -> Tuple[float, int]:
    metric = _group_reasoning_metric(samples) if sort_by_reasoning else float("inf")
    first_position = min(sample.position for sample in samples)
    return (metric, first_position)


def build_mode_records(
    groups: Dict[str, List[Sample]], sort_by_reasoning: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: _group_order_key(item[1], sort_by_reasoning),
    )
    mode_records: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in MODE_OUTPUT_ORDER}
    for _, samples in ordered_groups:
        for sample in sorted(samples, key=lambda entry: entry.position):
            mode_label = MODE_LABELS.get(sample.mode)
            if not mode_label:
                continue
            sample_data = sample.data or {}
            reasoning = sample_data.get("reasoning")
            if not isinstance(reasoning, str) or not reasoning.strip():
                continue
            record: Dict[str, Any] = {
                "id": sample_data.get("id"),
                "generation_index": sample_data.get("generation_index"),
                "prompt": sample_data.get("prompt"),
                "predict": sample_data.get("predict"),
                "label": sample_data.get("label"),
                "reasoning": reasoning,
                "reasoning_tokens": sample.reasoning_tokens,
            }
            mode_records[mode_label].append(record)
    return mode_records


def write_bucket(
    groups: Dict[str, List[Sample]],
    destinations: Dict[str, Path],
    dry_run: bool,
    sort_by_reasoning: bool = False,
) -> Dict[str, int]:
    mode_records = build_mode_records(groups, sort_by_reasoning=sort_by_reasoning)
    written_counts: Dict[str, int] = {}
    for mode in MODE_OUTPUT_ORDER:
        records = mode_records.get(mode, [])
        written_counts[mode] = len(records)
        destination = destinations[mode]
        if dry_run:
            continue
        if not records:
            if destination.exists():
                destination.unlink()
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as dst:
            for record in records:
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
    return written_counts


def summarize_bucket(groups: Dict[str, List[Sample]]) -> Tuple[int, int]:
    count = len(groups)
    return count, count


def gather_reasoning_values(buckets: Dict[str, Dict[str, List[Sample]]]) -> Dict[str, List[float]]:
    values: Dict[str, List[float]] = {}
    for name, groups in buckets.items():
        per_bucket: List[float] = []
        for samples in groups.values():
            per_bucket.extend(
                sample.reasoning_tokens
                for sample in samples
                if sample.reasoning_tokens is not None
            )
        values[name] = per_bucket
    return values


def normalize_ratios(ratios: Dict[str, float]) -> Dict[str, float]:
    cleaned = {key: max(0.0, value) for key, value in ratios.items()}
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("At least one ratio value must be positive.")
    return {key: value / total for key, value in cleaned.items()}


def compute_target_counts(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    if total <= 0:
        return {key: 0 for key in ratios}

    raw = {key: ratios[key] * total for key in ratios}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    if remainder > 0:
        # distribute remainder based on largest fractional parts
        fractional = sorted(
            raw.items(), key=lambda item: item[1] - math.floor(item[1]), reverse=True
        )
        for key, _ in fractional:
            if remainder <= 0:
                break
            counts[key] += 1
            remainder -= 1
    elif remainder < 0:
        # trim counts if rounding exceeded total (shouldn't happen but guard)
        for key in ratios:
            if remainder == 0:
                break
            reduction = min(counts[key], abs(remainder))
            counts[key] -= reduction
            remainder += reduction
    return counts


def _group_sort_key(samples: List[Sample]) -> Tuple[float, int]:
    values = [sample.reasoning_tokens for sample in samples if sample.reasoning_tokens is not None]
    avg = statistics.fmean(values) if values else float("inf")
    position = min(sample.position for sample in samples)
    return (avg, position)


def extract_even_groups(source: Dict[str, List[Sample]], amount: int) -> List[Tuple[str, List[Sample]]]:
    if amount <= 0 or not source:
        return []
    amount = min(amount, len(source))
    ordered = sorted(source.items(), key=lambda item: _group_sort_key(item[1]))
    total = len(ordered)
    selected_indices: List[int] = []
    for k in range(amount):
        target = (k + 0.5) / amount * total
        idx = max(0, min(total - 1, int(round(target - 1))))
        while idx in selected_indices and idx < total - 1:
            idx += 1
        while idx in selected_indices and idx > 0:
            idx -= 1
        selected_indices.append(idx)
    selected_indices = sorted(set(selected_indices))
    while len(selected_indices) < amount:
        for idx in range(total):
            if idx not in selected_indices:
                selected_indices.append(idx)
                if len(selected_indices) == amount:
                    break
    selected_indices = sorted(selected_indices[:amount])
    chosen: List[Tuple[str, List[Sample]]] = []
    for idx in selected_indices:
        key, samples = ordered[idx]
        chosen.append((key, samples))
    for key, _ in chosen:
        source.pop(key, None)
    return chosen


def rebalance_buckets(
    buckets: Dict[str, Dict[str, List[Sample]]], target_counts: Dict[str, int]
) -> List[str]:
    log_messages: List[str] = []

    for bucket_name in ("val", "grpo_train"):
        current = len(buckets.get(bucket_name, {}))
        target = target_counts.get(bucket_name, 0)
        deficit = target - current
        if deficit <= 0:
            continue
        moved_groups = extract_even_groups(buckets["sft_train"], deficit)
        if not moved_groups:
            log_messages.append(
                f"[WARN] Not enough SFT ids to fill '{bucket_name}' target "
                f"(needed {deficit}, moved 0)."
            )
            continue
        for sample_id, samples in moved_groups:
            buckets[bucket_name][sample_id] = samples
        log_messages.append(
            f"[INFO] Rebalanced {len(moved_groups)} ids from SFT to {bucket_name}."
        )
        remaining_deficit = target - len(buckets[bucket_name])
        if remaining_deficit > 0:
            log_messages.append(
                f"[WARN] Still short {remaining_deficit} ids for '{bucket_name}' target."
            )

    sft_target = target_counts.get("sft_train", len(buckets["sft_train"]))
    if len(buckets["sft_train"]) < sft_target:
        log_messages.append(
            "[WARN] SFT split below target after rebalancing; consider adjusting ratios."
        )
    return log_messages


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
    colors = {
        "sft_train": "#3A7BD5",
        "grpo_train": "#F4A261",
        "val": "#2A9D8F",
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    bin_centers = [
        (edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)
    ]
    per_bucket_peaks: Dict[str, int] = {}
    for bucket_name in ("sft_train", "grpo_train", "val"):
        values = reasoning_values.get(bucket_name, [])
        if not values:
            continue
        counts = histogram_counts(values, edges)
        per_bucket_peaks[bucket_name] = max(counts) if counts else 0
        ax.step(
            bin_centers,
            counts,
            where="mid",
            label=bucket_name,
            color=colors.get(bucket_name, "#555555"),
            linewidth=2,
        )
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
    colors = {
        "sft_train": "#3A7BD5",
        "grpo_train": "#F4A261",
        "val": "#2A9D8F",
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False

    for bucket_name in ("sft_train", "grpo_train", "val"):
        values = reasoning_values.get(bucket_name, [])
        if not values:
            continue
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        ys = [(idx + 1) / n for idx in range(n)]
        ax.plot(
            sorted_vals,
            ys,
            label=bucket_name,
            color=colors.get(bucket_name, "#555555"),
            linewidth=2,
        )
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


def process_dataset(
    dataset: str,
    input_root: Path,
    output_root: Path,
    limit: Optional[int],
    dry_run: bool,
    ratios: Dict[str, float],
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

    samples_by_id, mode_counts = collect_samples(files)
    if not samples_by_id:
        print(f"[{dataset}] No valid samples collected from {len(files)} files.")
        return

    buckets = categorize_samples(samples_by_id)
    normalized_ratios = normalize_ratios(ratios)
    total_ids = len(samples_by_id)
    target_counts = compute_target_counts(total_ids, normalized_ratios)
    rebalance_logs = rebalance_buckets(buckets, target_counts)

    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    outputs = {
        split: {
            mode: output_dir / f"{dataset}_{split}_{mode}.jsonl"
            for mode in MODE_OUTPUT_ORDER
        }
        for split in ("sft_train", "grpo_train", "val")
    }

    print(f"[{dataset}] Mode counts: " + ", ".join(f"{mode}={count}" for mode, count in mode_counts.items()))
    for message in rebalance_logs:
        print(message)

    bucket_record_counts: Dict[str, int] = {}
    bucket_id_stats: Dict[str, int] = {}
    per_bucket_rows: List[List[Any]] = []
    for bucket_name, groups in buckets.items():
        record_count, id_total = summarize_bucket(groups)
        sort_by_reasoning = bucket_name == "sft_train"
        written_counts = write_bucket(
            groups,
            outputs[bucket_name],
            dry_run=dry_run,
            sort_by_reasoning=sort_by_reasoning,
        )
        bucket_record_counts[bucket_name] = record_count
        bucket_id_stats[bucket_name] = id_total
        per_bucket_rows.append(
            [
                bucket_name,
                record_count,
                id_total,
                "0.00%",
            ]
        )
        total_entries_written = sum(written_counts.values())
        destinations = ", ".join(
            f"{mode}:{outputs[bucket_name][mode]}"
            for mode in MODE_OUTPUT_ORDER
        )
        print(
            f"  - {bucket_name}: ids={id_total} samples={record_count} "
            f"entries={total_entries_written} -> {destinations if not dry_run else '(dry run)'}"
        )

    total_records = sum(bucket_record_counts.values())
    if total_records:
        for row in per_bucket_rows:
            name = row[0]
            count = bucket_record_counts.get(name, 0)
            row[3] = f"{(count / total_records * 100):.2f}%"

        headers = ["Split", "Samples", "Unique IDs", "Ratio"]
        print(f"[{dataset}] Split summary:")
        print(tabulate(per_bucket_rows, headers=headers, tablefmt="grid"))

        print(f"[{dataset}] Bucket ratios (by sample count):")
        for bucket_name in ("sft_train", "grpo_train", "val"):
            count = bucket_record_counts.get(bucket_name, 0)
            ratio = (count / total_records * 100) if total_records else 0.0
            target = target_counts.get(bucket_name, 0)
            target_ratio = (target / total_ids * 100) if total_ids else 0.0
            print(
                f"  {bucket_name}: {count} samples ({ratio:.2f}% of {total_records}) "
                f"| target {target} ({target_ratio:.2f}%)"
            )

    reasoning_values = gather_reasoning_values(buckets)
    combined_values = [value for values in reasoning_values.values() for value in values]
    if combined_values:
        edges = build_histogram_edges(combined_values)
        counts = histogram_counts(combined_values, edges)
        peak = max(counts) if counts else 0
        y_limit = peak * 1.05 if peak else 1.0
        plot_path = save_histogram_overlay(
            dataset=dataset,
            reasoning_values=reasoning_values,
            edges=edges,
            output_dir=plot_dir,
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

    print(
        f"[{dataset}] Completed split: "
        f"SFT ids={bucket_id_stats.get('sft_train', 0)}, "
        f"GRPO ids={bucket_id_stats.get('grpo_train', 0)}, "
        f"VAL ids={bucket_id_stats.get('val', 0)}"
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
        ratios={
            "sft_train": args.ratio_sft,
            "grpo_train": args.ratio_grpo,
            "val": args.ratio_val,
        },
    )


if __name__ == "__main__":
    main()
