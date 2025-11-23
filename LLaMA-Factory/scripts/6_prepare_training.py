#!/usr/bin/env python3
"""Convert step-5 split outputs into prompt/response training JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
SPLIT_NAMES = ("sft_train", "grpo_train", "val")
MODE_NAMES = ("low", "medium", "high")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = BASE_DIR / "new_dataset" / "5_split"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "new_dataset" / "6_train"

SPECIAL_TOKENS = [
    "<|start|>user<|message|>",
    "<|end|><|start|>assistant",
    "<|start|>assistant",
    "<|end|>",
    "<|return|>",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transform the merged JSONL files from step 5 into prompt/response training format "
            "with <think> reasoning blocks per difficulty."
        )
    )
    parser.add_argument("dataset", choices=DATASET_CHOICES, help="Dataset name.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing the step-5 split files (default: new_dataset/5_split).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where the training JSONL files will be written (default: new_dataset/6_train).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing output files.",
    )
    return parser.parse_args()


def clean_prompt(prompt: str) -> str:
    text = prompt
    for token in SPECIAL_TOKENS:
        text = text.replace(token, " ")
    return " ".join(text.split())


def build_training_entry(
    record: Dict[str, object], mode: str, split: str
) -> Optional[Tuple[Dict[str, object], int]]:
    prompt_raw = record.get("prompt")
    if not isinstance(prompt_raw, str) or not prompt_raw.strip():
        return None
    prompt = clean_prompt(prompt_raw)
    predict = record.get("predict") if isinstance(record.get("predict"), str) else ""
    reasoning = record.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        if split == "grpo_train":
            reasoning = record.get("label")
        if not isinstance(reasoning, str) or not reasoning.strip():
            return None
    reasoning_clean = reasoning.strip()
    response = f"<think>{reasoning_clean}</think> {predict}".strip()
    reasoning_tokens_raw = record.get("reasoning_tokens")
    if isinstance(reasoning_tokens_raw, (int, float)):
        length = int(reasoning_tokens_raw)
    else:
        length = len(reasoning_clean.split())
    record_mode_raw = record.get("mode")
    if isinstance(record_mode_raw, str) and record_mode_raw.strip():
        normalized_mode = record_mode_raw.strip().lower()
    else:
        normalized_mode = mode
    if normalized_mode not in MODE_NAMES and normalized_mode != "grpo":
        normalized_mode = mode
    entry: Dict[str, object] = {
        "prompt": prompt,
        "response": response,
        "mode": normalized_mode,
    }
    if split != "grpo_train":
        entry["reasoning_tokens"] = length
    instruction = record.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        entry["instruction"] = instruction.strip()
    label_value = record.get("label")
    if isinstance(label_value, str) and label_value.strip():
        entry["label"] = label_value.strip()
    return entry, length


def derive_paths(
    dataset: str, split: str, mode: Optional[str], input_root: Path, output_root: Path
) -> Tuple[Path, Path]:
    if split == "grpo_train":
        input_path = input_root / dataset / f"{dataset}_{split}.jsonl"
        output_path = output_root / dataset / f"{dataset}_{split}_think.jsonl"
    else:
        input_path = input_root / dataset / f"{dataset}_{split}_{mode}.jsonl"
        output_path = output_root / dataset / f"{dataset}_{split}_{mode}_think.jsonl"
    return input_path, output_path


def process_file(
    input_path: Path, output_path: Path, split: str, mode: str, dry_run: bool
) -> Tuple[int, int, List[int]]:
    if not input_path.is_file():
        print(f"[WARN] Missing input file: {input_path}")
        if not dry_run and output_path.exists():
            output_path.unlink()
        return 0, 0, []

    total_records = 0
    total_entries = 0
    reasoning_lengths: List[int] = []

    writer = None
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = output_path.open("w", encoding="utf-8")

    with input_path.open("r", encoding="utf-8") as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            total_records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            built = build_training_entry(record, mode=mode, split=split)
            if built is None:
                continue
            entry, length = built
            total_entries += 1
            reasoning_lengths.append(length)
            if writer is not None:
                writer.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if writer is not None:
        writer.close()

    if dry_run and output_path.exists():
        output_path.unlink()

    return total_records, total_entries, reasoning_lengths


def process_dataset(
    dataset: str,
    input_root: Path,
    output_root: Path,
    dry_run: bool,
) -> None:
    dataset_input_dir = input_root / dataset
    if not dataset_input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {dataset_input_dir}")

    dataset_output_dir = output_root / dataset
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    mode_rows = []
    total_records = 0
    total_entries = 0

    path_rows = []
    reasoning_stats: Dict[Tuple[str, str], List[int]] = {}
    split_record_counts = {split: 0 for split in SPLIT_NAMES}
    split_entry_counts = {split: 0 for split in SPLIT_NAMES}

    tasks: List[Tuple[str, Optional[str]]] = []
    for split in ("sft_train", "val"):
        tasks.extend((split, mode) for mode in MODE_NAMES)
    tasks.append(("grpo_train", "grpo"))

    for split, mode in tasks:
        input_path, output_path = derive_paths(dataset, split, mode, input_root, output_root)
        records, entries, lengths = process_file(
            input_path, output_path, split=split, mode=mode or "grpo", dry_run=dry_run
        )
        split_record_counts[split] += records
        split_entry_counts[split] += entries
        mode_rows.append([split, mode, entries])
        reasoning_stats[(split, mode or "grpo")] = lengths
        path_rows.append(
            [
                split,
                mode,
                str(input_path),
                str(output_path) if not dry_run else "(dry run)",
            ]
        )
        total_records += records
        total_entries += entries

    for split in SPLIT_NAMES:
        rows.append(
            [
                split,
                split_record_counts[split],
                split_entry_counts[split],
            ]
        )

    headers = ["Split", "Records", "Training Entries", "Entry Ratio"]
    print(f"[{dataset}] Training conversion summary:")
    ratio_rows = []
    for split, records, entries in rows:
        ratio = entries / total_entries * 100 if total_entries else 0.0
        ratio_rows.append([split, records, entries, f"{ratio:.2f}%"])
    print(tabulate(ratio_rows, headers=headers, tablefmt="grid"))

    mode_headers = ["Split", "Mode", "Entries"]
    print(f"[{dataset}] Entries per mode:")
    print(tabulate(mode_rows, headers=mode_headers, tablefmt="grid"))

    print(f"[{dataset}] File paths:")
    for split, mode, input_path, output_path in path_rows:
        print(f"  {split} / {mode}:")
        print(f"    input  -> {input_path}")
        print(f"    output -> {output_path}")
    print(
        f"[{dataset}] Total source records={total_records}, "
        f"total training entries={total_entries}."
    )
    if not dry_run:
        save_reasoning_plots(dataset, dataset_output_dir, reasoning_stats)
    else:
        print("[INFO] Dry run mode; no files were written.")


def save_reasoning_plots(
    dataset: str,
    output_dir: Path,
    stats: Dict[Tuple[str, str], List[int]],
) -> None:
    if plt is None:
        print("[WARN] matplotlib not available; skipping reasoning length plots.")
        return
    plot_dir = output_dir / "reasoning_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        split_modes = {mode for (split_key, mode) in stats.keys() if split_key == split}
        for mode in sorted(split_modes):
            lengths = stats.get((split, mode), [])
            if not lengths:
                continue
            sorted_lengths = sorted(lengths)
            x_values = list(range(1, len(sorted_lengths) + 1))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x_values, sorted_lengths, color="#3A7BD5", linewidth=1.2)
            ax.set_title(f"{dataset} {split} {mode}: reasoning length order")
            ax.set_xlabel("Sample (sorted)")
            ax.set_ylabel("Reasoning length (tokens)")
            ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
            fig.tight_layout()
            path = plot_dir / f"{dataset}_{split}_{mode}_lengths.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)

        if split == "sft_train":
            fig, ax = plt.subplots(figsize=(8, 4))
            plotted = False
            colors = {"low": "#2A9D8F", "medium": "#F4A261", "high": "#E76F51"}
            for mode in MODE_NAMES:
                lengths = stats.get((split, mode), [])
                if not lengths:
                    continue
                sorted_lengths = sorted(lengths)
                x_values = list(range(1, len(sorted_lengths) + 1))
                ax.plot(
                    x_values,
                    sorted_lengths,
                    label=mode,
                    linewidth=1.2,
                    color=colors.get(mode, "#555555"),
                )
                plotted = True
            if plotted:
                ax.set_title(f"{dataset} {split}: reasoning length order (all modes)")
                ax.set_xlabel("Sample (sorted)")
                ax.set_ylabel("Reasoning length (tokens)")
                ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
                ax.legend()
                fig.tight_layout()
                path = plot_dir / f"{dataset}_{split}_all_lengths.png"
                fig.savefig(path, dpi=200)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    process_dataset(
        dataset=args.dataset,
        input_root=args.input_root,
        output_root=args.output_root,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
