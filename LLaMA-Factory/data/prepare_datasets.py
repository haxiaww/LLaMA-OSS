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
    """
    Clear instruction: reasoning goes internal, final answer is clean boxed format.
    """
    lines = [
        question.strip(),
        "",
        "Please reason step by step, and put your final answer within \\boxed{}.",
        "Do NOT include explanations or reasoning in the final answer - only the numeric value in \\boxed{}."
    ]
    return "\n".join(lines)


def process_gsm8k(outdir: str):
    print("[GSM8K] Loading openai/gsm8k ...")
    ds = load_dataset("openai/gsm8k", 'main')
    for split in ("train", "test"):
        if split not in ds:
            continue
        rows = []
        for ex in ds[split]:
            q = ex.get("question", "").strip()
            a = ex.get("answer", "").strip()
            rows.append({
                "prompt": build_prompt_gsm8k(q),
                "response": a,
                "meta": {"dataset": "gsm8k", "split": split}
            })
        out_path = os.path.join(outdir, f"gsm8k_{split}_alpaca.jsonl")
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


def build_prompt_logiqa(context: str, query: str, options: List[str]) -> str:
    """
    Clear format: reasoning internal, final output is just the letter in boxed format.
    """
    opts = options or []
    while len(opts) < 4:
        opts.append("")
    
    lines = [
        f"Context: {context.strip()}",
        "",
        f"Question: {query.strip()}",
        "",
        "Options:",
        f"A. {opts[0]}",
        f"B. {opts[1]}",
        f"C. {opts[2]}",
        f"D. {opts[3]}",
        "",
        "Please reason step by step, and put your final answer as a single letter (A/B/C/D) within \\boxed{}.",
        "Do NOT include explanations or reasoning in the final answer - only the letter in \\boxed{}."
    ]
    return "\n".join(lines)


def load_logiqa_dataset():
    try:
        print("[LogiQA] Trying parquet-converted revision (refs/convert/parquet) ...")
        return load_dataset("lucasmccabe/logiqa", revision="refs/convert/parquet")
    except Exception as e1:
        print(f"[LogiQA] Parquet conversion not available ({e1}). Falling back to trust_remote_code=True ...")
        return load_dataset("lucasmccabe/logiqa", trust_remote_code=True)


def process_logiqa(outdir: str):
    print("[LogiQA] Loading lucasmccabe/logiqa ...")
    ds = load_logiqa_dataset()
    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        rows = []
        for ex in ds[split]:
            context = ex.get("context", "") or ""
            query = ex.get("query", "") or ""
            options = ex.get("answers") or ex.get("options") or []
            correct = normalize_correct_option(ex.get("correct_option"))
            rows.append({
                "prompt": build_prompt_logiqa(context, query, options),
                "response": correct,
                "meta": {"dataset": "logiqa", "split": split, "raw_correct_option": ex.get("correct_option")}
            })
        out_path = os.path.join(outdir, f"logiqa_{split}_alpaca.jsonl")
        write_jsonl(out_path, rows)
        print(f"[LogiQA] Wrote {len(rows)} rows to {out_path}")


# ---------------------
# Competition Math (AIME-style)
# ---------------------
def build_prompt_compmath(problem: str) -> str:
    """
    AIME style: reasoning internal, final answer is clean boxed expression.
    """
    lines = [
        problem.strip(),
        "",
        "Please reason step by step, and put your final answer within \\boxed{}.",
        "Do NOT include explanations or reasoning in the final answer - only the mathematical expression or value in \\boxed{}."
    ]
    return "\n".join(lines)


def process_competition_math(outdir: str):
    print("[CompetitionMath] Loading qwedsacf/competition_math ...")
    ds = load_dataset("qwedsacf/competition_math")
    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        rows = []
        for ex in ds[split]:
            prob = ex.get("problem", "") or ex.get("question", "") or ""
            sol = ex.get("solution", "") or ex.get("answer", "") or ""
            meta = {
                "dataset": "competition_math",
                "split": split,
            }
            for key in ("level", "type", "source"):
                if key in ex:
                    meta[key] = ex[key]
            rows.append({
                "prompt": build_prompt_compmath(prob),
                "response": sol,
                "meta": meta
            })
        if rows:
            out_path = os.path.join(outdir, f"competition_math_{split}_alpaca.jsonl")
            write_jsonl(out_path, rows)
            print(f"[CompetitionMath] Wrote {len(rows)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for GPT-OSS inference (clean final answers)"
    )
    parser.add_argument("--outdir", type=str, default="data", help="Output folder for JSONL files.")
    parser.add_argument("--gsm8k", action="store_true", help="Process GSM8K (openai/gsm8k).")
    parser.add_argument("--logiqa", action="store_true", help="Process LogiQA (lucasmccabe/logiqa).")
    parser.add_argument("--comp-math", action="store_true", help="Process Competition Math (qwedsacf/competition_math).")
    parser.add_argument("--all", action="store_true", help="Process all datasets.")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    
    print("\n" + "="*70)
    print("GPT-OSS Dataset Preparation")
    print("="*70)
    print("Prompt strategy:")
    print("  ✓ 'Please reason step by step' - triggers internal CoT reasoning")
    print("  ✓ 'put your final answer within \\boxed{}' - clear output marker")
    print("  ✓ 'Do NOT include explanations in final answer' - clean output")
    print("")
    print("Expected behavior:")
    print("  • Reasoning → goes to 'reasoning_content' field (harmony format)")
    print("  • Final answer → goes to 'content' field as \\boxed{value} only")
    print("="*70 + "\n")

    if args.all or args.gsm8k:
        process_gsm8k(args.outdir)

    if args.all or args.logiqa:
        process_logiqa(args.outdir)

    if args.all or args.comp_math:
        process_competition_math(args.outdir)

    print("\n" + "="*70)
    print("Done! Example outputs:")
    print("")
    print("GSM8K:")
    print("  reasoning_content: 'Let me solve this step by step...'")
    print("  content: '\\boxed{42}'")
    print("")
    print("LogiQA:")
    print("  reasoning_content: 'Analyzing each option...'")
    print("  content: '\\boxed{D}'")
    print("")
    print("CompMath:")
    print("  reasoning_content: 'Using integration by parts...'")
    print("  content: '\\boxed{\\frac{\\pi}{4}}'")
    print("="*70 + "\n")
    

if __name__ == "__main__":
    main()
