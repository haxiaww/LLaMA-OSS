#!/usr/bin/env python3
"""
Intelligent Curriculum Learning Filter with GRPO Hard Examples.

Logic:
1. Merge low/med/high modes together
2. For each unique problem (by id):
   - If LOW mode got it correct → Use LOW (easiest solution)
   - Else if MED mode got it correct → Use MED
   - Else if HIGH mode got it correct → Use HIGH
   - Else (all failed) → Save to GRPO dataset (hard examples)
3. Deduplicate by id and reasoning tokens
4. Generate summary statistics
"""

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tabulate import tabulate
except ImportError:
    print("Missing 'tabulate'. Install with: pip install tabulate")
    import sys
    sys.exit(1)

DATASET_CHOICES = ("gsm8k", "compmath")
MODE_ORDER = ["low", "medium", "high"]  # Priority order

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = BASE_DIR / "new_dataset" / "1_fbl"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "new_dataset" / "curriculum"


@dataclass
class Sample:
    """Represents a single sample with metadata."""
    record: Dict[str, Any]
    sample_id: str
    mode: str
    reasoning_tokens: float
    is_correct: bool  # Whether predict == label
    line: str


def get_sample_id(record: Dict[str, Any]) -> Optional[str]:
    """Extract sample ID from record."""
    # Try _meta first
    if "_meta" in record and "id" in record["_meta"]:
        return str(record["_meta"]["id"])
    # Try id field
    if "id" in record:
        return str(record["id"])
    # Try combined_index
    if "combined_index" in record:
        return str(record["combined_index"])
    return None


def is_correct_prediction(record: Dict[str, Any]) -> bool:
    """Check if prediction matches label (already filtered by filter script)."""
    # If the file comes from 1_fbl, it's already filtered for correctness
    # But we can double-check
    predict = record.get("predict", "")
    label = record.get("label", "")
    return bool(predict and label)  # Both exist means it was kept by filter


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


def read_samples(file_path: Path) -> List[Sample]:
    """Read samples from JSONL file."""
    samples = []
    mode = extract_mode(file_path.name)
    
    if not mode:
        print(f"Warning: Could not extract mode from {file_path.name}")
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
                is_correct = is_correct_prediction(record)
                
                samples.append(Sample(
                    record=record,
                    sample_id=sample_id,
                    mode=mode,
                    reasoning_tokens=reasoning_tokens,
                    is_correct=is_correct,
                    line=line + "\n"
                ))
            except Exception as e:
                print(f"Warning: Error reading line: {e}")
                continue
    
    return samples


def deduplicate_by_reasoning_tokens(samples: List[Sample]) -> List[Sample]:
    """
    For samples with same ID, keep the one closest to mean reasoning_tokens.
    """
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.sample_id].append(sample)
    
    deduplicated = []
    
    for sample_id, group in groups.items():
        if len(group) == 1:
            deduplicated.append(group[0])
        else:
            # Calculate mean
            mean_tokens = statistics.mean(s.reasoning_tokens for s in group)
            # Keep closest to mean
            closest = min(group, key=lambda s: abs(s.reasoning_tokens - mean_tokens))
            deduplicated.append(closest)
    
    return deduplicated


def write_samples(samples: List[Sample], output_path: Path) -> int:
    """Write samples to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample.record, ensure_ascii=False) + '\n')
    
    return len(samples)


def generate_statistics(samples: List[Sample], name: str) -> Dict[str, Any]:
    """Generate statistics for a set of samples."""
    if not samples:
        return {
            "name": name,
            "total": 0,
            "unique_ids": 0,
            "by_mode": {},
            "reasoning_tokens": {}
        }
    
    unique_ids = len(set(s.sample_id for s in samples))
    
    by_mode = defaultdict(int)
    for sample in samples:
        by_mode[sample.mode] += 1
    
    tokens = [s.reasoning_tokens for s in samples]
    
    return {
        "name": name,
        "total": len(samples),
        "unique_ids": unique_ids,
        "by_mode": dict(by_mode),
        "reasoning_tokens": {
            "mean": statistics.mean(tokens) if tokens else 0,
            "median": statistics.median(tokens) if tokens else 0,
            "std": statistics.stdev(tokens) if len(tokens) > 1 else 0,
            "min": min(tokens) if tokens else 0,
            "max": max(tokens) if tokens else 0,
        }
    }


def print_summary(stats_list: List[Dict[str, Any]]):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    for stats in stats_list:
        print(f"\n{stats['name']}:")
        print(f"  Total samples: {stats['total']}")
        print(f"  Unique IDs: {stats['unique_ids']}")
        
        if stats['by_mode']:
            print(f"  By mode:")
            for mode in ["low", "medium", "high"]:
                if mode in stats['by_mode']:
                    count = stats['by_mode'][mode]
                    pct = (count / stats['total'] * 100) if stats['total'] > 0 else 0
                    print(f"    {mode:8s}: {count:6d} ({pct:5.1f}%)")
        
        if stats['reasoning_tokens']:
            rt = stats['reasoning_tokens']
            print(f"  Reasoning tokens:")
            print(f"    Mean:   {rt['mean']:.2f}")
            print(f"    Median: {rt['median']:.2f}")
            print(f"    Std:    {rt['std']:.2f}")
            print(f"    Range:  [{rt['min']:.0f}, {rt['max']:.0f}]")


def process_dataset(
    dataset: str,
    input_root: Path,
    output_root: Path,
) -> None:
    """Process a single dataset."""
    
    print("=" * 80)
    print(f"Processing {dataset.upper()}")
    print("=" * 80)
    
    # Step 1: Read FILTERED files (correct predictions)
    filtered_dir = input_root / dataset
    rejected_dir = input_root / "rejected" / dataset
    
    if not filtered_dir.exists():
        print(f"Error: Filtered directory not found: {filtered_dir}")
        return
    
    print("\n" + "-" * 80)
    print("STEP 1: Reading FILTERED samples (correct predictions)")
    print("-" * 80)
    
    filtered_samples = []
    for mode in ["low", "med", "medium", "high"]:
        pattern = f"*{mode}*fbl.jsonl"
        matching_files = list(filtered_dir.glob(pattern))
        
        for file_path in matching_files:
            print(f"Reading: {file_path.name}")
            samples = read_samples(file_path)
            print(f"  Loaded {len(samples)} correct samples")
            filtered_samples.extend(samples)
    
    # Get all unique IDs that were solved correctly
    solved_ids = set(s.sample_id for s in filtered_samples)
    print(f"\nTotal filtered samples: {len(filtered_samples)}")
    print(f"Unique problems solved: {len(solved_ids)}")
    
    # Step 2: Read REJECTED files (incorrect predictions)
    print("\n" + "-" * 80)
    print("STEP 2: Reading REJECTED samples (incorrect predictions)")
    print("-" * 80)
    
    rejected_samples = []
    if rejected_dir.exists():
        for mode in ["low", "med", "medium", "high"]:
            pattern = f"*{mode}*rejected.jsonl"
            matching_files = list(rejected_dir.glob(pattern))
            
            for file_path in matching_files:
                print(f"Reading: {file_path.name}")
                samples = read_samples(file_path)
                print(f"  Loaded {len(samples)} rejected samples")
                rejected_samples.extend(samples)
    
    # Get all unique IDs that were attempted but failed
    rejected_ids = set(s.sample_id for s in rejected_samples)
    print(f"\nTotal rejected samples: {len(rejected_samples)}")
    print(f"Unique problems rejected: {len(rejected_ids)}")
    
    # Step 3: Find hard examples (problems that ALL modes failed)
    print("\n" + "-" * 80)
    print("STEP 3: Finding hard examples (GRPO candidates)")
    print("-" * 80)
    
    # Hard examples = IDs that appear in rejected but NEVER in filtered
    hard_ids = rejected_ids - solved_ids
    print(f"Problems where ALL modes failed: {len(hard_ids)}")
    
    # Get all rejected samples for hard IDs
    grpo_samples = [s for s in rejected_samples if s.sample_id in hard_ids]
    
    # Step 4: Apply curriculum learning to SOLVED problems
    print("\n" + "-" * 80)
    print("STEP 4: Applying curriculum learning to SOLVED problems")
    print("-" * 80)
    
    # Group filtered samples by ID and mode
    solved_groups = defaultdict(lambda: {"low": [], "medium": [], "high": []})
    for sample in filtered_samples:
        solved_groups[sample.sample_id][sample.mode].append(sample)
    
    curriculum_samples = []
    
    for sample_id, modes in solved_groups.items():
        selected = None
        
        # Try LOW first (easiest)
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
    
    print(f"Curriculum samples selected: {len(curriculum_samples)}")
    
    # Count by mode
    mode_counts = defaultdict(int)
    for s in curriculum_samples:
        mode_counts[s.record.get("curriculum_source", s.mode)] += 1
    
    print("  Distribution:")
    for mode in ["low", "medium", "high"]:
        count = mode_counts[mode]
        pct = (count / len(curriculum_samples) * 100) if curriculum_samples else 0
        print(f"    {mode:8s}: {count:6d} ({pct:5.1f}%)")
    
    # Step 5: Deduplicate GRPO samples
    print("\n" + "-" * 80)
    print("STEP 5: Deduplicating GRPO samples")
    print("-" * 80)
    
    grpo_dedup = deduplicate_by_reasoning_tokens(grpo_samples)
    print(f"  Before: {len(grpo_samples)} samples")
    print(f"  After:  {len(grpo_dedup)} samples")
    
    # Step 6: Write outputs
    output_dir = output_root / dataset
    curriculum_path = output_dir / f"{dataset}_curriculum.jsonl"
    grpo_path = output_dir / f"{dataset}_grpo_hard.jsonl"
    
    print("\n" + "-" * 80)
    print("STEP 6: Writing outputs")
    print("-" * 80)
    
    curriculum_count = write_samples(curriculum_samples, curriculum_path)
    print(f"  Curriculum: {curriculum_path}")
    print(f"    {curriculum_count} samples")
    
    grpo_count = write_samples(grpo_dedup, grpo_path)
    print(f"  GRPO (hard): {grpo_path}")
    print(f"    {grpo_count} samples")
    
    # Step 7: Generate statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    
    stats_curriculum = generate_statistics(curriculum_samples, f"{dataset} - Curriculum (CL)")
    stats_grpo = generate_statistics(grpo_dedup, f"{dataset} - GRPO (Hard)")
    
    print_summary([stats_curriculum, stats_grpo])
    
    # Create summary table
    summary_rows = [
        ["Total problems attempted", len(solved_ids) + len(hard_ids), "-"],
        ["Problems solved (any mode)", len(solved_ids), f"{len(solved_ids)/(len(solved_ids)+len(hard_ids))*100:.1f}%"],
        ["Curriculum samples", len(curriculum_samples), f"100.0%"],
        ["Problems failed (all modes)", len(hard_ids), f"{len(hard_ids)/(len(solved_ids)+len(hard_ids))*100:.1f}%"],
        ["GRPO samples (dedup)", len(grpo_dedup), "-"],
    ]
    
    print("\n" + "=" * 80)
    print(f"{dataset.upper()} - FINAL SUMMARY")
    print("=" * 80)
    print(tabulate(summary_rows, headers=["Category", "Count", "Percentage"], tablefmt="grid"))
    
    # Detailed breakdown
    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN")
    print("=" * 80)
    
    breakdown_rows = [
        ["Filtered (correct)", "", len(filtered_samples), len(solved_ids)],
        ["  Low mode", "✓", sum(1 for s in filtered_samples if s.mode == "low"), "-"],
        ["  Medium mode", "✓", sum(1 for s in filtered_samples if s.mode == "medium"), "-"],
        ["  High mode", "✓", sum(1 for s in filtered_samples if s.mode == "high"), "-"],
        ["", "", "", ""],
        ["Rejected (incorrect)", "", len(rejected_samples), len(rejected_ids)],
        ["  Low mode", "✗", sum(1 for s in rejected_samples if s.mode == "low"), "-"],
        ["  Medium mode", "✗", sum(1 for s in rejected_samples if s.mode == "medium"), "-"],
        ["  High mode", "✗", sum(1 for s in rejected_samples if s.mode == "high"), "-"],
        ["", "", "", ""],
        ["Hard (failed all modes)", "✗✗✗", len(grpo_samples), len(hard_ids)],
        ["", "", "", ""],
        ["Curriculum output", "📚", len(curriculum_samples), len(curriculum_samples)],
        ["  From low", "", mode_counts["low"], f"{mode_counts['low']/len(curriculum_samples)*100:.1f}%"],
        ["  From medium", "", mode_counts["medium"], f"{mode_counts['medium']/len(curriculum_samples)*100:.1f}%"],
        ["  From high", "", mode_counts["high"], f"{mode_counts['high']/len(curriculum_samples)*100:.1f}%"],
        ["", "", "", ""],
        ["GRPO output", "🔥", len(grpo_dedup), len(grpo_dedup)],
    ]
    
    print(tabulate(breakdown_rows, headers=["Category", "Status", "Samples", "Unique IDs"], tablefmt="grid"))


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent curriculum learning filter with GRPO hard examples"
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        help="Dataset to process"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Input root directory (default: new_dataset/1_fbl)"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root directory (default: new_dataset/curriculum)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all datasets"
    )
    
    args = parser.parse_args()
    
    if args.all:
        datasets = DATASET_CHOICES
    elif args.dataset:
        datasets = [args.dataset]
    else:
        print("Error: Specify --dataset or --all")
        return
    
    for dataset in datasets:
        process_dataset(dataset, args.input_root, args.output_root)
        print("\n")


if __name__ == "__main__":
    main()