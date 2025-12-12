#!/usr/bin/env python3
"""
Complete pipeline: Post-process, move, filter, and merge datasets.

Usage:
    python3 1_process_all.py --compmath-raw compmath_train_low.jsonl --gsm8k-raw gsm8k_train_low_raw.jsonl
    python3 1_process_all.py --compmath-raw compmath_train_med.jsonl --gsm8k-raw gsm8k_train_med.jsonl
    python3 1_process_all.py --compmath-raw compmath_train_high.jsonl --gsm8k-raw gsm8k_train_high.jsonl
"""

import argparse
import json
import re
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional, Tuple

try:
    from tabulate import tabulate
except ImportError:
    print("Missing dependency 'tabulate'. Install it with: pip install tabulate")
    sys.exit(1)

# =============================================================================
# POST-PROCESSING FUNCTIONS
# =============================================================================

def clean_special_tokens(text: str) -> str:
    """Remove special tokens from text."""
    text = re.sub(r'<\|end\|><\|start\|>assistant<\|channel\|>final<\|message\|>', '', text)
    text = re.sub(r'<\|end\|><\|return\|>', '', text)
    text = re.sub(r'<\|channel\|>[^<]*<\|message\|>', '', text)
    text = re.sub(r'<\|[^>]+\|>', '', text)
    return text.strip()


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the last \\boxed{} answer from text."""
    boxed_pattern = r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}'
    matches = list(re.finditer(boxed_pattern, text))
    
    if matches:
        last_match = matches[-1]
        return f"\\boxed{{{last_match.group(1)}}}"
    
    return None


def extract_thinking_and_answer_compmath(response: str) -> Tuple[str, str]:
    """Extract thinking and answer for CompMath."""
    text = clean_special_tokens(response)
    
    boxed_pattern = r'\\boxed\{'
    matches = list(re.finditer(boxed_pattern, text))
    
    if not matches:
        return text.strip(), ""
    
    last_boxed_pos = matches[-1].start()
    thinking = text[:last_boxed_pos].strip()
    answer = extract_boxed_answer(text)
    
    if not answer:
        answer = ""
    
    thinking = re.sub(r'(?:the answer is|therefore|thus|hence)[:\s]*$', '', thinking, flags=re.IGNORECASE).strip()
    
    return thinking, answer


def extract_answer_gsm8k(text: str) -> Optional[str]:
    """Extract answer from text for GSM8K (#### or \\boxed{})."""
    boxed_pattern = r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}'
    boxed_matches = list(re.finditer(boxed_pattern, text))
    if boxed_matches:
        last_match = boxed_matches[-1]
        return f"\\boxed{{{last_match.group(1)}}}"
    
    hash_pattern = r'####\s*([^\n]+)'
    hash_match = re.search(hash_pattern, text)
    if hash_match:
        answer = hash_match.group(1).strip()
        return f"#### {answer}"
    
    return None


def normalize_to_hash_format(text: str) -> Optional[str]:
    """Normalize answer to #### format for GSM8K."""
    boxed_pattern = r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}'
    boxed_matches = list(re.finditer(boxed_pattern, text))
    if boxed_matches:
        answer = boxed_matches[-1].group(1).strip()
        return f"#### {answer}"
    
    hash_pattern = r'####\s*([^\n]+)'
    hash_match = re.search(hash_pattern, text)
    if hash_match:
        answer = hash_match.group(1).strip()
        return f"#### {answer}"
    
    stripped = text.strip()
    if stripped and (stripped.replace('.', '').replace('-', '').isdigit() or 
                     any(c.isdigit() for c in stripped)):
        return f"#### {stripped}"
    
    return None


def extract_thinking_and_answer_gsm8k(response: str) -> Tuple[str, str]:
    """Extract thinking and answer for GSM8K."""
    text = clean_special_tokens(response)
    
    boxed_pattern = r'\\boxed\{'
    boxed_matches = list(re.finditer(boxed_pattern, text))
    
    if boxed_matches:
        last_boxed_pos = boxed_matches[-1].start()
        thinking = text[:last_boxed_pos].strip()
        answer = extract_answer_gsm8k(text)
        if not answer:
            answer = ""
        
        thinking = re.sub(r'(?:the answer is|therefore|thus|hence)[:\s]*$', '', thinking, flags=re.IGNORECASE).strip()
        return thinking, answer
    
    hash_pattern = r'####'
    hash_match = re.search(hash_pattern, text)
    
    if hash_match:
        hash_pos = hash_match.start()
        thinking = text[:hash_pos].strip()
        answer = extract_answer_gsm8k(text)
        if not answer:
            answer = ""
        
        thinking = re.sub(r'(?:the answer is|therefore|thus|hence)[:\s]*$', '', thinking, flags=re.IGNORECASE).strip()
        return thinking, answer
    
    return text.strip(), ""


def process_compmath_file(input_path: Path, output_path: Path) -> Tuple[int, int]:
    """Process CompMath file."""
    processed = 0
    with_thinking = 0
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                
                response = record.get('response', '')
                if response:
                    thinking, answer = extract_thinking_and_answer_compmath(response)
                    
                    if thinking:
                        formatted_response = f"<think>{thinking}</think> {answer}"
                    else:
                        formatted_response = answer
                    
                    record['response'] = formatted_response
                    record['predict'] = answer
                    
                    label = record.get('label', '')
                    if label:
                        label_answer = extract_boxed_answer(clean_special_tokens(label))
                        if label_answer:
                            record['label'] = label_answer
                    
                    if thinking:
                        record['reasoning_tokens'] = len(thinking) / 4.0
                        with_thinking += 1
                    else:
                        record['reasoning_tokens'] = 0.0
                
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                processed += 1
                
            except Exception as e:
                print(f"Warning: Error processing line: {e}", file=sys.stderr)
                continue
    
    return processed, with_thinking


def process_gsm8k_file(input_path: Path, output_path: Path) -> Tuple[int, int]:
    """Process GSM8K file."""
    processed = 0
    with_thinking = 0
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                
                response = record.get('response', '')
                if response:
                    thinking, answer = extract_thinking_and_answer_gsm8k(response)
                    
                    normalized_answer = normalize_to_hash_format(answer) if answer else ""
                    
                    if thinking and normalized_answer:
                        formatted_response = f"<think>{thinking}</think> {normalized_answer}"
                    elif normalized_answer:
                        formatted_response = normalized_answer
                    else:
                        formatted_response = answer
                    
                    record['response'] = formatted_response
                    record['predict'] = normalized_answer if normalized_answer else answer
                    
                    label = record.get('label', '')
                    if label:
                        normalized_label = normalize_to_hash_format(clean_special_tokens(label))
                        if normalized_label:
                            record['label'] = normalized_label
                    
                    if thinking:
                        record['reasoning_tokens'] = len(thinking) / 4.0
                        with_thinking += 1
                    else:
                        record['reasoning_tokens'] = 0.0
                
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                processed += 1
                
            except Exception as e:
                print(f"Warning: Error processing line: {e}", file=sys.stderr)
                continue
    
    return processed, with_thinking


# =============================================================================
# FILTERING FUNCTIONS
# =============================================================================

BOXED_PATTERN = re.compile(r"\\boxed\{([^}]+)\}")
HASHED_PATTERN = re.compile(r"####\s*([^\n]+)")


def _clean_candidate(text: str) -> Optional[str]:
    """Normalize extracted answers."""
    cleaned = text.strip().rstrip(".").strip()
    return cleaned or None


def _is_prediction_complete(pred: Optional[str]) -> bool:
    """Check if prediction is complete (has either \\boxed{} or ####)."""
    if not isinstance(pred, str):
        return False
    stripped = pred.strip()
    if not stripped:
        return False
    
    # Check for \boxed{} format
    boxed_matches = BOXED_PATTERN.findall(stripped)
    if len(boxed_matches) > 0 and any(m.strip() for m in boxed_matches):
        return True
    
    # Check for #### format (GSM8K)
    hash_matches = HASHED_PATTERN.findall(stripped)
    if len(hash_matches) > 0 and any(m.strip() for m in hash_matches):
        return True
    
    return False


def extract_answer_for_filter(value: Any) -> Optional[str]:
    """Extract answer for filtering."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    
    matches = BOXED_PATTERN.findall(stripped)
    for match in matches:
        candidate = _clean_candidate(match)
        if candidate:
            return candidate.replace("<|end|><|return|>", "")
    
    hash_match = HASHED_PATTERN.search(stripped)
    if hash_match:
        candidate = _clean_candidate(hash_match.group(1))
        if candidate:
            return candidate.replace("<|end|><|return|>", "")
    return None


def extract_label_answer(dataset: str, label_raw: Any) -> Optional[str]:
    """Extract label answer using dataset-specific rules."""
    if dataset in {"gsm8k", "compmath"}:
        return extract_answer_for_filter(label_raw)
    return None


def filter_dataset(
    dataset: str,
    input_path: Path,
    output_path: Path,
    rejected_path: Optional[Path],
) -> Tuple[int, int]:
    """Filter samples by correctness."""
    kept = 0
    rejected = 0
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rejected_path:
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, "r", encoding="utf-8") as src, \
         open(output_path, "w", encoding="utf-8") as dst, \
         (open(rejected_path, "w", encoding="utf-8") if rejected_path else nullcontext()) as rejected_file:
        
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                if rejected_file is not None:
                    rejected_file.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                    rejected += 1
                continue
            
            predict_raw = sample.get("predict")
            label_raw = sample.get("label")
            
            if not _is_prediction_complete(predict_raw):
                if rejected_file is not None:
                    rejected_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    rejected += 1
                continue
            
            predict_answer = extract_answer_for_filter(predict_raw)
            label_answer = extract_label_answer(dataset, label_raw)
            
            if (
                predict_answer is not None
                and label_answer is not None
                and predict_answer == label_answer
            ):
                dst.write(json.dumps(sample, ensure_ascii=False) + "\n")
                kept += 1
            elif rejected_file is not None:
                rejected_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                rejected += 1
    
    return kept, rejected


def merge_files(file_list: list[Path], output_path: Path) -> int:
    """Merge multiple JSONL files into one."""
    total_lines = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as fout:
        for file_path in file_list:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as fin:
                    for line in fin:
                        fout.write(line)
                        total_lines += 1
    
    return total_lines


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Complete pipeline: process, filter, and merge datasets")
    parser.add_argument("--compmath-raw", type=str, help="CompMath raw input file")
    parser.add_argument("--gsm8k-raw", type=str, help="GSM8K raw input file")
    parser.add_argument("--output-dir", type=str, default="./new_dataset", help="Output directory")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merging filtered files")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "0_raw"
    filtered_dir = output_dir / "1_fbl"
    rejected_dir = filtered_dir / "rejected"
    merged_dir = output_dir / "2_merged"
    
    # Create directories
    for d in [raw_dir, filtered_dir, rejected_dir, merged_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    results = []
    filtered_files = {"compmath": [], "gsm8k": []}
    
    # ==========================================================================
    # Process CompMath
    # ==========================================================================
    if args.compmath_raw:
        print("=" * 70)
        print("STEP 1: Processing CompMath")
        print("=" * 70)
        
        compmath_input = Path(args.compmath_raw)
        if not compmath_input.exists():
            print(f"Error: CompMath file not found: {compmath_input}")
        else:
            # Post-process
            stem = compmath_input.stem.replace("_raw", "")
            processed_path = raw_dir / "compmath" / f"{stem}.jsonl"
            
            print(f"Post-processing: {compmath_input.name} -> {processed_path.name}")
            processed, with_thinking = process_compmath_file(compmath_input, processed_path)
            print(f"  Processed: {processed}, With <think>: {with_thinking} ({with_thinking/processed*100:.1f}%)")
            
            # Filter
            filtered_path = filtered_dir / "compmath" / f"{stem}_fbl.jsonl"
            rejected_path = rejected_dir / "compmath" / f"{stem}_rejected.jsonl"
            
            print(f"Filtering: {processed_path.name}")
            kept, rejected_count = filter_dataset("compmath", processed_path, filtered_path, rejected_path)
            
            results.append(["compmath", stem, processed, kept, rejected_count])
            filtered_files["compmath"].append(filtered_path)
            
            print(f"  Kept: {kept}, Rejected: {rejected_count}")
            print()
    
    # ==========================================================================
    # Process GSM8K
    # ==========================================================================
    if args.gsm8k_raw:
        print("=" * 70)
        print("STEP 2: Processing GSM8K")
        print("=" * 70)
        
        gsm8k_input = Path(args.gsm8k_raw)
        if not gsm8k_input.exists():
            print(f"Error: GSM8K file not found: {gsm8k_input}")
        else:
            # Post-process
            stem = gsm8k_input.stem.replace("_raw", "")
            processed_path = raw_dir / "gsm8k" / f"{stem}.jsonl"
            
            print(f"Post-processing: {gsm8k_input.name} -> {processed_path.name}")
            processed, with_thinking = process_gsm8k_file(gsm8k_input, processed_path)
            print(f"  Processed: {processed}, With <think>: {with_thinking} ({with_thinking/processed*100:.1f}%)")
            
            # Filter
            filtered_path = filtered_dir / "gsm8k" / f"{stem}_fbl.jsonl"
            rejected_path = rejected_dir / "gsm8k" / f"{stem}_rejected.jsonl"
            
            print(f"Filtering: {processed_path.name}")
            kept, rejected_count = filter_dataset("gsm8k", processed_path, filtered_path, rejected_path)
            
            results.append(["gsm8k", stem, processed, kept, rejected_count])
            filtered_files["gsm8k"].append(filtered_path)
            
            print(f"  Kept: {kept}, Rejected: {rejected_count}")
            print()
    
    # ==========================================================================
    # Summary Table
    # ==========================================================================
    if results:
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        headers = ["Dataset", "File", "Total", "Kept", "Rejected"]
        print(tabulate(results, headers=headers, tablefmt="grid"))
        print()
    
    # ==========================================================================
    # Merge Filtered Files
    # ==========================================================================
    if not args.skip_merge and (filtered_files["compmath"] or filtered_files["gsm8k"]):
        print("=" * 70)
        print("STEP 3: Merging Filtered Files")
        print("=" * 70)
        
        # Merge CompMath
        if filtered_files["compmath"]:
            compmath_merged = merged_dir / "compmath_all_filtered.jsonl"
            total = merge_files(filtered_files["compmath"], compmath_merged)
            print(f"CompMath: Merged {len(filtered_files['compmath'])} files -> {compmath_merged.name} ({total} samples)")
        
        # Merge GSM8K
        if filtered_files["gsm8k"]:
            gsm8k_merged = merged_dir / "gsm8k_all_filtered.jsonl"
            total = merge_files(filtered_files["gsm8k"], gsm8k_merged)
            print(f"GSM8K:    Merged {len(filtered_files['gsm8k'])} files -> {gsm8k_merged.name} ({total} samples)")
        
        # Merge all datasets together
        all_filtered = filtered_files["compmath"] + filtered_files["gsm8k"]
        if all_filtered:
            all_merged = merged_dir / "all_datasets_filtered.jsonl"
            total = merge_files(all_filtered, all_merged)
            print(f"ALL:      Merged {len(all_filtered)} files -> {all_merged.name} ({total} samples)")
        
        print()
    
    # ==========================================================================
    # Final Output Paths
    # ==========================================================================
    print("=" * 70)
    print("OUTPUT LOCATIONS")
    print("=" * 70)
    print(f"Processed files:  {raw_dir}/")
    print(f"Filtered files:   {filtered_dir}/")
    print(f"Rejected files:   {rejected_dir}/")
    if not args.skip_merge:
        print(f"Merged files:     {merged_dir}/")
    print()
    print("Done! ✓")


if __name__ == "__main__":
    main()