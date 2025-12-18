#!/usr/bin/env python3
"""
Create 3 separate SFT datasets + 1 GRPO dataset.

For each mode (low/medium/high):
- Merge GSM8K and CompMath
- Filter by length (Q25-Q75 range)
- Deduplicate by ID (keep median-length response)
- Clean prompts
- Save SFT dataset

For GRPO:
- Collect all filtered-out samples (outside Q25-Q75)
- Format for GRPO training
- Clean prompts
- Deduplicate by prompt

Output: 
- 3 SFT files (low_sft.jsonl, medium_sft.jsonl, high_sft.jsonl)
- 1 GRPO file (grpo.jsonl)

Usage:
    python3 create_all_datasets.py --input-dir ./new_dataset/0_raw --output-dir ./final
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set
import sys

try:
    from tabulate import tabulate
except ImportError:
    print("Missing 'tabulate'. Install with: pip install tabulate")
    sys.exit(1)

# Length thresholds per dataset and mode (using Q25-Q75 interquartile range)
# Based on actual tokenized statistics
LENGTH_THRESHOLDS = {
    "compmath": {
        "high": {"min": 237.0, "max": 534.0},     # Q25: 237.0, Q75: 534.0
        "low": {"min": 56.0, "max": 121.0},       # Q25: 56.0,  Q75: 121.0
        "medium": {"min": 133.0, "max": 292.0},   # Q25: 133.0, Q75: 292.0
    },
    "gsm8k": {
        "high": {"min": 183.0, "max": 405.0},     # Q25: 183.0, Q75: 405.0
        "low": {"min": 45.0, "max": 78.0},        # Q25: 45.0,  Q75: 78.0
        "medium": {"min": 98.0, "max": 194.0},    # Q25: 98.0,  Q75: 194.0
    },
}


def clean_prompt(prompt: str) -> str:
    """Remove chat template formatting from prompt."""
    # Remove system message
    system_pattern = r'<\|start\|>system<\|message\|>.*?<\|end\|><\|start\|>user<\|message\|>'
    cleaned = re.sub(system_pattern, '', prompt, flags=re.DOTALL)
    
    # Remove assistant suffix
    assistant_pattern = r'<\|end\|><\|start\|>assistant.*?$'
    cleaned = re.sub(assistant_pattern, '', cleaned, flags=re.DOTALL)
    
    return cleaned.strip()


def get_sample_id(record: Dict) -> str:
    """Extract unique sample ID from record."""
    dataset = record.get("dataset", "unknown")
    
    if "_meta" in record and "id" in record["_meta"]:
        sample_id = str(record["_meta"]["id"])
    elif "id" in record:
        sample_id = str(record["id"])
    elif "combined_index" in record:
        sample_id = str(record["combined_index"])
    else:
        prompt = record.get("prompt", "")
        sample_id = str(abs(hash(prompt)))
    
    return f"{dataset}_{sample_id}"


def get_prompt_hash(prompt: str) -> str:
    """Get hash of cleaned prompt for deduplication."""
    cleaned = clean_prompt(prompt)
    # Normalize whitespace
    normalized = ' '.join(cleaned.split())
    return str(hash(normalized))


def read_mode_files(input_dir: Path, mode: str, collect_filtered_out: bool = False) -> Tuple[List[Dict], List[Dict]]:
    """
    Read all files for a specific mode from both datasets.
    
    Args:
        input_dir: Directory containing dataset folders
        mode: Mode to read (low, medium, high)
        collect_filtered_out: If True, collect filtered-out samples for GRPO
        
    Returns:
        Tuple of (kept_records, filtered_out_records)
    """
    kept_records = []
    filtered_out_records = []
    
    for dataset in ["gsm8k", "compmath"]:
        dataset_dir = input_dir / dataset
        if not dataset_dir.exists():
            print(f"  Warning: {dataset_dir} not found")
            continue
        
        # Find files matching this mode
        all_files = list(dataset_dir.glob("*.jsonl"))
        
        for file_path in all_files:
            filename_lower = file_path.name.lower()
            
            # Skip rejected/filtered files
            if "rejected" in filename_lower or "fbl" in filename_lower:
                continue
            
            # Check if this file is for the current mode
            if mode == "low" and "low" not in filename_lower:
                continue
            elif mode == "medium" and not ("med" in filename_lower or "medium" in filename_lower):
                continue
            elif mode == "high" and "high" not in filename_lower:
                continue
            
            print(f"  Reading: {dataset}/{file_path.name}")
            
            # Get length threshold for this dataset and mode
            threshold_range = LENGTH_THRESHOLDS.get(dataset, {}).get(mode, {"min": 0, "max": 10000})
            min_threshold = threshold_range.get("min", 0)
            max_threshold = threshold_range.get("max", 10000)
            filtered_count = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        
                        # Filter by length threshold (Q25-Q75 range)
                        reasoning_tokens = record.get("reasoning_tokens", 0.0)
                        in_range = min_threshold <= reasoning_tokens <= max_threshold
                        
                        # Set metadata
                        record["mode"] = mode
                        if "dataset" not in record:
                            record["dataset"] = dataset
                        
                        # Clean prompt
                        if "prompt" in record:
                            record["prompt"] = clean_prompt(record["prompt"])
                        
                        if in_range:
                            # Keep for SFT
                            kept_records.append(record)
                        else:
                            # Filtered out - add to GRPO
                            filtered_count += 1
                            if collect_filtered_out:
                                filtered_out_records.append(record)
                        
                    except json.JSONDecodeError:
                        continue
            
            if filtered_count > 0:
                print(f"    Filtered {filtered_count} samples (outside [{min_threshold}, {max_threshold}] tokens)")
                if collect_filtered_out:
                    print(f"    Added to GRPO dataset")
    
    return kept_records, filtered_out_records


def deduplicate_by_id(records: List[Dict]) -> List[Dict]:
    """
    Deduplicate records by ID, keeping the one with median reasoning_tokens.
    
    Args:
        records: List of records
        
    Returns:
        Deduplicated list
    """
    # Group by ID
    id_to_records = defaultdict(list)
    for record in records:
        sample_id = get_sample_id(record)
        id_to_records[sample_id].append(record)
    
    # Select median for each ID
    deduplicated = []
    for sample_id, group in id_to_records.items():
        if len(group) == 1:
            deduplicated.append(group[0])
        else:
            # Sort by reasoning_tokens and pick median
            group.sort(key=lambda r: r.get("reasoning_tokens", 0.0))
            median_idx = len(group) // 2
            deduplicated.append(group[median_idx])
    
    return deduplicated


def deduplicate_by_prompt(records: List[Dict]) -> List[Dict]:
    """
    Deduplicate records by prompt hash.
    
    Args:
        records: List of records
        
    Returns:
        Deduplicated list
    """
    seen_prompts: Set[str] = set()
    deduplicated = []
    
    for record in records:
        prompt = record.get("prompt", "")
        prompt_hash = get_prompt_hash(prompt)
        
        if prompt_hash not in seen_prompts:
            seen_prompts.add(prompt_hash)
            deduplicated.append(record)
    
    return deduplicated


def format_grpo_record(record: Dict) -> Dict:
    """
    Format record for GRPO training.
    
    GRPO format:
    {
      "prompt": "cleaned prompt",
      "label": "ground truth answer",

      "dataset": "gsm8k"
    }
    """
    return {
        "prompt": record.get("prompt", ""),  # Already cleaned
        "label": record.get("label", ""),
        "dataset": record.get("dataset", ""),
    }


def process_mode(
    mode: str,
    input_dir: Path,
    output_dir: Path,
    existing_grpo_path: Path = None,
) -> Tuple[int, int, Dict, int]:
    """
    Process one mode: merge datasets, filter, deduplicate.
    Also create GRPO dataset for this mode from filtered-out samples.
    
    Returns:
        Tuple of (total_records, final_sft_count, dataset_counts, grpo_count)
    """
    print("=" * 70)
    print(f"Processing Mode: {mode.upper()}")
    print("=" * 70)
    
    # Read all files for this mode
    kept_records, filtered_out = read_mode_files(input_dir, mode, collect_filtered_out=True)
    print(f"\nTotal records loaded: {len(kept_records)}")
    print(f"Filtered out for GRPO: {len(filtered_out)}")
    
    # === SFT Processing ===
    if not kept_records:
        print("Warning: No records found for SFT!")
        sft_count = 0
        dataset_counts = {}
    else:
        # Count by dataset before deduplication
        before_counts = defaultdict(int)
        for record in kept_records:
            before_counts[record.get("dataset", "unknown")] += 1
        
        print(f"\nSFT - Before deduplication:")
        for dataset, count in sorted(before_counts.items()):
            print(f"  {dataset:10s}: {count:6d}")
        
        # Deduplicate by ID
        deduplicated = deduplicate_by_id(kept_records)
        print(f"\nSFT - After deduplication: {len(deduplicated)} samples")
        
        # Count by dataset after deduplication
        dataset_counts = defaultdict(int)
        for record in deduplicated:
            dataset_counts[record.get("dataset", "unknown")] += 1
        
        print(f"\nSFT - Final counts:")
        for dataset, count in sorted(dataset_counts.items()):
            pct = (count / len(deduplicated) * 100) if deduplicated else 0
            print(f"  {dataset:10s}: {count:6d} ({pct:5.1f}%)")
        
        # Sort by dataset for consistent ordering (GSM8K first, then CompMath)
        deduplicated.sort(key=lambda r: (r.get("dataset", ""), r.get("prompt", "")))
        
        # Write SFT output
        sft_path = output_dir / f"{mode}_sft.jsonl"
        sft_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(sft_path, 'w', encoding='utf-8') as f:
            for record in deduplicated:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"\nSFT output: {sft_path}")
        sft_count = len(deduplicated)
        dataset_counts = dict(dataset_counts)
    
    # === GRPO Processing ===
    print(f"\n{'-'*70}")
    print(f"GRPO for {mode.upper()} mode")
    print(f"{'-'*70}")
    
    grpo_samples = filtered_out.copy()
    print(f"Filtered-out samples: {len(grpo_samples)}")
    
    # Add existing GRPO if provided
    if existing_grpo_path and existing_grpo_path.exists():
        print(f"Reading existing GRPO: {existing_grpo_path}")
        with open(existing_grpo_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # Clean prompt if not already
                    if "prompt" in record:
                        record["prompt"] = clean_prompt(record["prompt"])
                    grpo_samples.append(record)
                except json.JSONDecodeError:
                    continue
        print(f"After adding existing: {len(grpo_samples)}")
    
    # Deduplicate GRPO by prompt
    print(f"Deduplicating by prompt...")
    deduplicated_grpo = deduplicate_by_prompt(grpo_samples)
    duplicates = len(grpo_samples) - len(deduplicated_grpo)
    print(f"  Before: {len(grpo_samples)}")
    print(f"  After:  {len(deduplicated_grpo)}")
    print(f"  Removed: {duplicates} duplicates")
    
    # Format for GRPO
    formatted_grpo = [format_grpo_record(record) for record in deduplicated_grpo]
    
    # Count by dataset
    grpo_dataset_counts = defaultdict(int)
    for record in formatted_grpo:
        grpo_dataset_counts[record.get("dataset", "unknown")] += 1
    
    print(f"\nGRPO - By dataset:")
    for dataset, count in sorted(grpo_dataset_counts.items()):
        pct = (count / len(formatted_grpo) * 100) if formatted_grpo else 0
        print(f"  {dataset:10s}: {count:6d} ({pct:5.1f}%)")
    
    # Write GRPO output
    grpo_path = output_dir / f"grpo_{mode}.jsonl"
    grpo_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(grpo_path, 'w', encoding='utf-8') as f:
        for record in formatted_grpo:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\nGRPO output: {grpo_path}")
    grpo_count = len(formatted_grpo)
    
    return len(kept_records), sft_count, dataset_counts, grpo_count


def main():
    parser = argparse.ArgumentParser(
        description="Create 3 SFT datasets + 3 GRPO datasets"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./new_dataset/0_raw"),
        help="Input directory (default: ./0_raw)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./final"),
        help="Output directory (default: ./final)"
    )
    parser.add_argument(
        "--existing-grpo",
        type=Path,
        help="Path to existing GRPO dataset (will be added to ALL 3 GRPO files)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CREATE 3 SFT DATASETS + 3 GRPO DATASETS")
    print("=" * 70)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    if args.existing_grpo:
        print(f"Existing GRPO: {args.existing_grpo}")
        print("  (will be added to ALL 3 GRPO files)")
    print("=" * 70)
    print()
    
    # Process each mode - each gets the full existing GRPO
    sft_results = []
    grpo_results = []
    
    for mode in ["low", "medium", "high"]:
        total, sft_count, dataset_counts, grpo_count = process_mode(
            mode, 
            args.input_dir, 
            args.output_dir,
            existing_grpo_path=args.existing_grpo
        )
        
        sft_results.append([
            mode.capitalize(),
            total,
            sft_count,
            dataset_counts.get("gsm8k", 0),
            dataset_counts.get("compmath", 0),
        ])
        
        grpo_results.append([
            mode.capitalize(),
            grpo_count,
        ])
        
        print()
    
    # Print SFT summary
    print("\n" + "=" * 70)
    print("SFT SUMMARY")
    print("=" * 70)
    
    headers = ["Mode", "Loaded", "Final", "GSM8K", "CompMath"]
    print(tabulate(sft_results, headers=headers, tablefmt="grid"))
    
    # Print GRPO summary
    print("\n" + "=" * 70)
    print("GRPO SUMMARY")
    print("=" * 70)
    
    headers = ["Mode", "Total Samples"]
    print(tabulate(grpo_results, headers=headers, tablefmt="grid"))
    
    # Calculate totals
    total_sft = sum(r[2] for r in sft_results)
    total_grpo_unique = sum(r[1] for r in grpo_results)  # Note: includes duplicates across modes
    
    print(f"\nTotal SFT samples: {total_sft}")
    print(f"Note: GRPO totals include duplicates (original GRPO in all 3 files)")
    
    # Print output files
    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    
    print("\nSFT Datasets:")
    for mode in ["low", "medium", "high"]:
        output_file = args.output_dir / f"{mode}_sft.jsonl"
        if output_file.exists():
            size = output_file.stat().st_size / 1024 / 1024  # MB
            print(f"  {output_file} ({size:.1f} MB)")
    
    print("\nGRPO Datasets:")
    for mode in ["low", "medium", "high"]:
        grpo_file = args.output_dir / f"grpo_{mode}.jsonl"
        if grpo_file.exists():
            size = grpo_file.stat().st_size / 1024 / 1024  # MB
            print(f"  {grpo_file} ({size:.1f} MB)")
    
    print("\n" + "=" * 70)
    print("TRAINING READY!")
    print("=" * 70)
    print("SFT Datasets (Q25-Q75 range):")
    print("  • low_sft.jsonl    - Concise reasoning")
    print("  • medium_sft.jsonl - Moderate reasoning")
    print("  • high_sft.jsonl   - Detailed reasoning")
    print("\nGRPO Datasets (original + mode-specific filtered):")
    print("  • grpo_low.jsonl    = Original GRPO + Low filtered-out")
    print("  • grpo_medium.jsonl = Original GRPO + Medium filtered-out")
    print("  • grpo_high.jsonl   = Original GRPO + High filtered-out")
    print("\nNote: All 3 GRPO files contain the original GRPO data")
    print("      Each also includes its mode's filtered-out samples")
    print("\nDone! ✓")


if __name__ == "__main__":
    main()