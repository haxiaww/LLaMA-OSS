#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_datasets.py - Optimized for GPT-OSS with clean final answers
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional

try:
    from datasets import load_dataset
except Exception as e:
    raise SystemExit("This script requires the 'datasets' package. Install via: pip install datasets")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------
# GSM8K - Math Problems
# ---------------------
def build_prompt_gsm8k(question: str) -> str:
    return question.strip()


def process_gsm8k(outdir: str):
    print("[GSM8K] Loading openai/gsm8k ...")
    ds = load_dataset("openai/gsm8k", 'main')
    for split in ("train",):
        if split not in ds:
            continue
        rows = []
        for ex in ds[split]:
            q = ex.get("question", "").strip()
            a = ex.get("answer", "").strip()
            reasoning, answer = a.split("####")[0].strip(), a.split("####")[1].strip()
            rows.append({
                "prompt": build_prompt_gsm8k(q),
                "response": "<think>" + reasoning + "</think> \\boxed{" + answer + "}",
            })
        out_path = os.path.join(outdir, f"gsm8k_origin_{split}_alpaca.jsonl")
        write_jsonl(out_path, rows)
        print(f"[GSM8K] Wrote {len(rows)} rows to {out_path}")


# ---------------------
# LogiQA - Logic Reasoning
# ---------------------
LETTER = ["A", "B", "C", "D"]

def normalize_correct_option(co: Any) -> str:
    if isinstance(co, int):
        if 0 <= co < len(LETTER):
            return LETTER[co]
        return str(co)
    if isinstance(co, str):
        s = co.strip().upper()
        if s in LETTER:
            return s
        try:
            idx = int(s)
            return LETTER[idx] if 0 <= idx < len(LETTER) else s
        except Exception:
            return s
    return str(co)


# def build_prompt_logiqa(context: str, query: str, options: List[str]) -> str:
#     """
#     Clear format: reasoning internal, final output is just the letter in boxed format.
#     """
#     opts = options or []
#     while len(opts) < 4:
#         opts.append("")
    
#     lines = [
#         f"Context: {context.strip()}",
#         "",
#         f"Question: {query.strip()}",
#         "",
#         "Options:",
#         f"A. {opts[0]}",
#         f"B. {opts[1]}",
#         f"C. {opts[2]}",
#         f"D. {opts[3]}",
#     ]
#     return "\n".join(lines)


# def load_logiqa_dataset():
#     try:
#         print("[LogiQA] Trying parquet-converted revision (refs/convert/parquet) ...")
#         return load_dataset("lucasmccabe/logiqa", revision="refs/convert/parquet")
#     except Exception as e1:
#         print(f"[LogiQA] Parquet conversion not available ({e1}). Falling back to trust_remote_code=True ...")
#         return load_dataset("lucasmccabe/logiqa", trust_remote_code=True)


# def process_logiqa(outdir: str):
#     print("[LogiQA] Loading lucasmccabe/logiqa ...")
#     ds = load_logiqa_dataset()
#     for split in ("train",):
#         if split not in ds:
#             continue
#         rows = []
#         for ex in ds[split]:
#             context = ex.get("context", "") or ""
#             query = ex.get("query", "") or ""
#             options = ex.get("answers") or ex.get("options") or []
#             correct = normalize_correct_option(ex.get("correct_option"))
#             rows.append({
#                 "prompt": build_prompt_logiqa(context, query, options),
#                 "response": correct,
#             })
#         out_path = os.path.join(outdir, f"logiqa_origin_{split}_alpaca.jsonl")
#         write_jsonl(out_path, rows)
#         print(f"[LogiQA] Wrote {len(rows)} rows to {out_path}")


# ---------------------
# Competition Math (AIME-style)
# ---------------------
def build_prompt_compmath(problem: str) -> str:
    return problem.strip()


def process_competition_math(outdir: str):
    print("[CompetitionMath] Loading DigitalLearningGmbH/MATH-lighteval")
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval")
    for split in ("train",):
        if split not in ds:
            continue
        rows = []
        problematic_indices = []
        total = 0
        for idx, ex in enumerate(ds[split]):
            total += 1
            prob = ex.get("problem", "") or ex.get("question", "") or ""
            sol = ex.get("solution", "") or ex.get("answer", "") or ""
            
            # Extract reasoning and answer
            # Competition math solutions typically have reasoning followed by \boxed{answer}
            import re
            boxed_match = re.search(r'\\boxed\{([^}]+)\}', sol)
            
            if boxed_match:
                # Extract the final answer from \boxed{}
                answer = boxed_match.group(1).strip()
                # Keep the entire solution (including \boxed{}) as reasoning
                reasoning = sol.strip()
            else:
                # No valid \boxed{} found - add with empty answer for manual review
                problematic_indices.append(idx)
                answer = ""
                reasoning = sol.strip() if sol else ""
            
            meta = {
                "dataset": "competition_math",
                "split": split,
            }
            for key in ("level", "type", "source"):
                if key in ex:
                    meta[key] = ex[key]
            rows.append({
                "prompt": build_prompt_compmath(prob),
                "response": "<think>" + reasoning + "</think> \\boxed{" + answer + "}",
                })
        if rows:
            out_path = os.path.join(outdir, f"competition_math_origin_{split}_alpaca.jsonl")
            write_jsonl(out_path, rows)
            print(f"[CompetitionMath] Wrote {len(rows)} rows to {out_path}")
            if problematic_indices:
                print(f"[CompetitionMath] ⚠️  Found {len(problematic_indices)} examples with missing/malformed \\boxed{{}}")
                print(f"[CompetitionMath] Indices needing manual review: {problematic_indices}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for GPT-OSS inference (clean final answers)"
    )
    parser.add_argument("--outdir", type=str, default="data", help="Output folder for JSONL files.")
    parser.add_argument("--gsm8k", action="store_true", help="Process GSM8K (openai/gsm8k).")
    # parser.add_argument("--logiqa", action="store_true", help="Process LogiQA (lucasmccabe/logiqa).")
    parser.add_argument("--comp-math", action="store_true", help="Process Competition Math (DigitalLearningGmbH/MATH-lighteval).")
    parser.add_argument("--all", action="store_true", help="Process all datasets.")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    
    print("\n" + "="*70)
    print("Origin Dataset Preparation")
    print("="*70)

    if args.all or args.gsm8k:
        process_gsm8k(args.outdir)

    # if args.all or args.logiqa:
    #     process_logiqa(args.outdir)

    if args.all or args.comp_math:
        process_competition_math(args.outdir)

    print("\n" + "="*70)
    print("Done!")
    

if __name__ == "__main__":
    main()
