"""
Complete Pipeline: Merge → Filter by tokens+mode → Split SFT/GRPO

Steps:
1. Merge GSM8K and CompMath (all low/med/high modes from 1_fbl)
2. Apply curriculum learning logic (prefer low → med → high)
3. Filter by reasoning token thresholds per mode
4. Deduplicate by ID
5. Split into SFT (curriculum) and GRPO (hard examples)
6. Format for training
"""

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

try:
    from tabulate import tabulate
except ImportError:
    print("Missing 'tabulate'. Install with: pip install tabulate")
    import sys
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = BASE_DIR / "new_dataset" / "1_fbl"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "new_dataset" / "final_merged"

# Reasoning token thresholds per mode (based on your statistics)
THRESHOLDS = {
    "low": {"min": 50, "max": 600},      # Mean: 220.83, keep reasonable range
    "medium": {"min": 200, "max": 1500}, # Mean: 622.55
    "high": {"min": 600, "max": 3000},   # Mean: 1415.05
}


@dataclass
class Sample:
    """Represents a single sample."""
    record: Dict
    sample_id: str
    mode: str
    reasoning_tokens: float
    dataset: str
    is_filtered: bool  # True if from filtered, False if from rejected


def get_sample_id(record: Dict) -> Optional[str]:
    """Extract sample ID."""
    if "_meta" in record and "id" in record["_meta"]:
        return str(record["_meta"]["id"])
    if "id" in record:
        return str(record["id"])
    if "combined_index" in record:
        return str(record["combined_index"])
    return None


def extract_mode(filename: str) -> Optional[str]:
    """Extract mode from filename."""
    filename_lower = filename.lower()
    if "low" in filename_lower:
        return "low"
    elif "med" in filename_lower or "medium" in filename_lower:
        return "medium"
    elif "high" in filename_lower:
        return "high"
    return None


def read_samples_from_file(file_path: Path, is_filtered: bool) -> List[Sample]:
    """Read samples from JSONL file."""
    samples = []
    mode = extract_mode(file_path.name)
    
    # Extract dataset from path
    dataset = None
    if "gsm8k" in str(file_path):
        dataset = "gsm8k"
    elif "compmath" in str(file_path):
        dataset = "compmath"
    
    if not mode or not dataset:
        print(f"Warning: Could not extract mode/dataset from {file_path.name}")
        return samples
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                sample_id = get_sample_id(record)
                
                if not sample_id:
                    continue
                
                reasoning_tokens = float(record.get("reasoning_tokens", 0.0))
                
                samples.append(Sample(
                    record=record,
                    sample_id=f"{dataset}_{sample_id}",  # Prefix with dataset
                    mode=mode,
                    reasoning_tokens=reasoning_tokens,
                    dataset=dataset,
                    is_filtered=is_filtered
                ))
            except Exception as e:
                continue
    
    return samples


def filter_by_reasoning_tokens(samples: List[Sample]) -> List[Sample]:
    """Filter samples by reasoning token thresholds per mode."""
    filtered = []
    
    for sample in samples:
        threshold = THRESHOLDS.get(sample.mode, {"min": 0, "max": 10000})
        
        if threshold["min"] <= sample.reasoning_tokens <= threshold["max"]:
            filtered.append(sample)
    
    return filtered


def apply_curriculum_logic(
    filtered_samples: List[Sample],
    rejected_samples: List[Sample]
) -> Tuple[List[Sample], List[Sample]]:
    """
    Apply curriculum learning logic:
    - For each unique ID, prefer LOW → MED → HIGH from filtered samples
    - IDs not in filtered = hard examples (GRPO)
    """
    # Get solved IDs
    solved_ids = set(s.sample_id for s in filtered_samples)
    rejected_ids = set(s.sample_id for s in rejected_samples)
    
    # Hard IDs = rejected but never solved
    hard_ids = rejected_ids - solved_ids
    
    # Group filtered samples by ID and mode
    solved_groups = defaultdict(lambda: {"low": [], "medium": [], "high": []})
    for sample in filtered_samples:
        solved_groups[sample.sample_id][sample.mode].append(sample)
    
    # Select best sample per ID (curriculum logic)
    curriculum_samples = []
    
    for sample_id, modes in solved_groups.items():
        selected = None
        
        # Try LOW first
        if modes["low"]:
            selected = min(modes["low"], key=lambda s: s.reasoning_tokens)
            selected.record["curriculum_source"] = "low"
        # Try MEDIUM
        elif modes["medium"]:
            selected = min(modes["medium"], key=lambda s: s.reasoning_tokens)
            selected.record["curriculum_source"] = "medium"
        # Try HIGH
        elif modes["high"]:
            selected = min(modes["high"], key=lambda s: s.reasoning_tokens)
            selected.record["curriculum_source"] = "high"
        
        if selected:
            curriculum_samples.append(selected)
    
    # Get hard examples
    grpo_samples = [s for s in rejected_samples if s.sample_id in hard_ids]
    
    return curriculum_samples, grpo_samples


def deduplicate_by_reasoning_tokens(samples: List[Sample]) -> List[Sample]:
    """Deduplicate: keep sample closest to mean reasoning_tokens per ID."""
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.sample_id].append(sample)
    
    deduplicated = []
    for sample_id, group in groups.items():
        if len(group) == 1:
            deduplicated.append(group[0])
        else:
            mean_tokens = statistics.mean(s.reasoning_tokens for s in group)
            closest = min(group, key=lambda s: abs(s.reasoning_tokens - mean_tokens))
            deduplicated.append(closest)
    
    return deduplicated


def format_sft_sample(sample: Sample) -> Dict:
    """Format sample for SFT training."""
    return {
        "prompt": sample.record.get("prompt", ""),
        "response": sample.record.get("response", ""),
        "mode": sample.mode,
        "instruction": sample.record.get("instruction", ""),
        "label": sample.record.get("label", ""),
        "dataset": sample.dataset,
        "reasoning_tokens": sample.reasoning_tokens,
        "curriculum_source": sample.record.get("curriculum_source", sample.mode),
    }


def format_grpo_sample(sample: Sample) -> Dict:
    """Format sample for GRPO training."""
    return {
        "prompt": sample.record.get("prompt", ""),
        "label": sample.record.get("label", ""),
        "mode": "hard",
        "dataset": sample.dataset,
    }


def write_samples(samples: List[Sample], output_path: Path, format_fn) -> int:
    """Write samples to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            formatted = format_fn(sample)
            f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
    
    return len(samples)


def generate_statistics(samples: List[Sample], name: str) -> Dict:
    """Generate statistics."""
    if not samples:
        return {
            "name": name,
            "total": 0,
            "by_dataset": {},
            "by_mode": {},
            "reasoning_tokens": {}
        }
    
    by_dataset = defaultdict(int)
    by_mode = defaultdict(int)
    
    for sample in samples:
        by_dataset[sample.dataset] += 1
        by_mode[sample.mode] += 1
    
    tokens = [s.reasoning_tokens for s in samples]
    
    return {
        "name": name,
        "total": len(samples),
        "by_dataset": dict(by_dataset),
        "by_mode": dict(by_mode),
        "reasoning_tokens": {
            "mean": statistics.mean(tokens) if tokens else 0,
            "median": statistics.median(tokens) if tokens else 0,
            "std": statistics.stdev(tokens) if len(tokens) > 1 else 0,
            "min": min(tokens) if tokens else 0,
            "max": max(tokens) if tokens else 0,
        }
    }


def print_statistics(stats_list: List[Dict]):
    """Print statistics."""
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    
    for stats in stats_list:
        print(f"\n{stats['name']}:")
        print(f"  Total: {stats['total']}")
        
        if stats['by_dataset']:
            print("  By dataset:")
            for ds in ["gsm8k", "compmath"]:
                if ds in stats['by_dataset']:
                    count = stats['by_dataset'][ds]
                    pct = (count / stats['total'] * 100) if stats['total'] > 0 else 0
                    print(f"    {ds:10s}: {count:6d} ({pct:5.1f}%)")
        
        if stats['by_mode']:
            print("  By mode:")
            for mode in ["low", "medium", "high", "hard"]:
                if mode in stats['by_mode']:
                    count = stats['by_mode'][mode]
                    pct = (count / stats['total'] * 100) if stats['total'] > 0 else 0
                    print(f"    {mode:10s}: {count:6d} ({pct:5.1f}%)")
        
        if stats['reasoning_tokens']:
            rt = stats['reasoning_tokens']
            print(f"  Reasoning tokens:")
            print(f"    Mean:   {rt['mean']:.2f}")
            print(f"    Median: {rt['median']:.2f}")
            print(f"    Std:    {rt['std']:.2f}")
            print(f"    Range:  [{rt['min']:.0f}, {rt['max']:.0f}]")


def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline: merge datasets, filter, split SFT/GRPO"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Input root (default: new_dataset/1_fbl)"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root (default: new_dataset/final_merged)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("STEP 1: Reading GSM8K and CompMath (Filtered + Rejected)")
    print("=" * 80)
    
    all_filtered = []
    all_rejected = []
    
    # Read GSM8K first
    for dataset in ["gsm8k", "compmath"]:
        print(f"\n{dataset.upper()}:")
        
        filtered_dir = args.input_root / dataset
        rejected_dir = args.input_root / "rejected" / dataset
        
        # Read filtered (correct)
        if filtered_dir.exists():
            for file_path in filtered_dir.glob("*fbl.jsonl"):
                samples = read_samples_from_file(file_path, is_filtered=True)
                all_filtered.extend(samples)
                print(f"  Filtered: {file_path.name} ({len(samples)} samples)")
        
        # Read rejected (incorrect)
        if rejected_dir.exists():
            for file_path in rejected_dir.glob("*rejected.jsonl"):
                samples = read_samples_from_file(file_path, is_filtered=False)
                all_rejected.extend(samples)
                print(f"  Rejected: {file_path.name} ({len(samples)} samples)")
    
    print(f"\n  Total filtered: {len(all_filtered)}")
    print(f"  Total rejected: {len(all_rejected)}")
    
    # Step 2: Filter by reasoning tokens
    print("\n" + "=" * 80)
    print("STEP 2: Filtering by Reasoning Token Thresholds")
    print("=" * 80)
    
    for mode, thresh in THRESHOLDS.items():
        print(f"  {mode:8s}: {thresh['min']:4d} - {thresh['max']:4d} tokens")
    
    filtered_by_tokens = filter_by_reasoning_tokens(all_filtered)
    rejected_by_tokens = filter_by_reasoning_tokens(all_rejected)
    
    print(f"\nFiltered samples: {len(all_filtered)} → {len(filtered_by_tokens)}")
    print(f"Rejected samples: {len(all_rejected)} → {len(rejected_by_tokens)}")
    
    # Step 3: Apply curriculum learning
    print("\n" + "=" * 80)
    print("STEP 3: Applying Curriculum Learning Logic")
    print("=" * 80)
    
    curriculum_samples, grpo_samples = apply_curriculum_logic(
        filtered_by_tokens,
        rejected_by_tokens
    )
    
    print(f"  Curriculum (solved): {len(curriculum_samples)}")
    print(f"  GRPO (hard):         {len(grpo_samples)}")
    
    # Step 4: Deduplicate
    print("\n" + "=" * 80)
    print("STEP 4: Deduplicating by Reasoning Tokens")
    print("=" * 80)
    
    curriculum_dedup = deduplicate_by_reasoning_tokens(curriculum_samples)
    grpo_dedup = deduplicate_by_reasoning_tokens(grpo_samples)
    
    print(f"  Curriculum: {len(curriculum_samples)} → {len(curriculum_dedup)}")
    print(f"  GRPO:       {len(grpo_samples)} → {len(grpo_dedup)}")
    
    # Step 5: Write outputs
    print("\n" + "=" * 80)
    print("STEP 5: Writing Final Datasets")
    print("=" * 80)
    
    sft_path = args.output_root / "merged_sft.jsonl"
    grpo_path = args.output_root / "merged_grpo.jsonl"
    
    sft_count = write_samples(curriculum_dedup, sft_path, format_sft_sample)
    grpo_count = write_samples(grpo_dedup, grpo_path, format_grpo_sample)
    
    print(f"\n  SFT:  {sft_path}")
    print(f"        {sft_count} samples")
    print(f"\n  GRPO: {grpo_path}")
    print(f"        {grpo_count} samples")
    
    # Step 6: Statistics
    stats_sft = generate_statistics(curriculum_dedup, "SFT (Curriculum Learning)")
    stats_grpo = generate_statistics(grpo_dedup, "GRPO (Hard Examples)")
    
    print_statistics([stats_sft, stats_grpo])
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    summary_rows = [
        ["SFT (Curriculum)", sft_count, f"{stats_sft['by_dataset'].get('gsm8k', 0)}", f"{stats_sft['by_dataset'].get('compmath', 0)}"],
        ["GRPO (Hard)", grpo_count, f"{stats_grpo['by_dataset'].get('gsm8k', 0)}", f"{stats_grpo['by_dataset'].get('compmath', 0)}"],
    ]
    
    print(tabulate(summary_rows, headers=["Type", "Total", "GSM8K", "CompMath"], tablefmt="grid"))
    
    print("\nDone! ✓")


if __name__ == "__main__":
    main()
