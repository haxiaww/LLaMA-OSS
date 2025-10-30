#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_datasets.py
-------------------
Download and preprocess the following datasets into Alpaca-style JSONL for inference:
- GSM8K:          openai/gsm8k            (splits: train, test)
- LogiQA:         lucasmccabe/logiqa       (splits: train, validation, test)
- CompetitionMath: qwedsacf/competition_math (splits: train; may vary)

Output folder: data/processed/
Each JSONL contains {"prompt": <str>, "response": <optional str>, "meta": <optional dict>} per line.

Requirements:
    pip install datasets

Usage examples:
    python prepare_datasets.py --all
    python prepare_datasets.py --gsm8k
    python prepare_datasets.py --logiqa
    python prepare_datasets.py --comp-math
    python prepare_datasets.py --outdir data/processed

Notes:
- "response" is kept to store ground truth labels/solutions for later evaluation.
- For inference, LLaMA-Factory will primarily use "prompt".
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
# GSM8K
# ---------------------
def build_prompt_gsm8k(question: str) -> str:
    lines = [
        "You are a careful math & logic assistant. Solve the following problem step by step, then provide the final numeric answer.",
        "",
        f"Problem: {question.strip()}",
        "",
        "Explain your reasoning, then output the final answer."
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
                "response": a,                     # keep full solution text (often includes '#### <final>')
                "meta": {"dataset": "gsm8k", "split": split}
            })
        out_path = os.path.join(outdir, f"gsm8k_{split}_alpaca.jsonl")
        write_jsonl(out_path, rows)
        print(f"[GSM8K] Wrote {len(rows)} rows to {out_path}")


# ---------------------
# LogiQA
# ---------------------
LETTER = ["A", "B", "C", "D"]

def normalize_correct_option(co: Any) -> str:
    """
    Map correct_option to a letter A/B/C/D. Some variants store int index (0..3) or the letter string.
    """
    if isinstance(co, int):
        if 0 <= co < len(LETTER):
            return LETTER[co]
        return str(co)
    if isinstance(co, str):
        s = co.strip().upper()
        # Could be "A"/"B"/"C"/"D" or possibly "0"/"1"/...
        if s in LETTER:
            return s
        try:
            idx = int(s)
            return LETTER[idx] if 0 <= idx < len(LETTER) else s
        except Exception:
            return s
    return str(co)


def build_prompt_logiqa(context: str, query: str, options: List[str]) -> str:
    # Some datasets use 'answers' or 'options'; ensure we handle both gracefully.
    opts = options or []
    while len(opts) < 4:
        opts.append("")
    lines = [
        "You are a careful logic QA assistant. Read the context and question, then explain your reasoning and pick the best option (A/B/C/D).",
        "",
        f"Context: {context.strip()}",
        f"Question: {query.strip()}",
        "Options:",
        f"A. {opts[0]}",
        f"B. {opts[1]}",
        f"C. {opts[2]}",
        f"D. {opts[3]}",
        "",
        "Explain your reasoning, then output the final answer as a single letter (A/B/C/D)."
    ]
    return "\n".join(lines)


def load_logiqa_dataset():
    """
    Handle Datasets>=3 where dataset scripts are not executed by default.
    Strategy:
      1) Try parquet-converted revision (no script execution needed)
      2) Fallback: trust_remote_code=True to allow running dataset script
    """
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
# Competition Math (MATH-like)
# ---------------------
def build_prompt_compmath(problem: str) -> str:
    lines = [
        "You are a careful math assistant. Solve the following competition math problem step by step and provide a final boxed answer.",
        "",
        f"Problem: {problem.strip()}",
        "",
        "Explain your reasoning, then provide the final answer."
    ]
    return "\n".join(lines)


def process_competition_math(outdir: str):
    print("[CompetitionMath] Loading qwedsacf/competition_math ...")
    ds = load_dataset("qwedsacf/competition_math")
    # Attempt common splits; if only "train" exists, process that.
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
            # Keep some metadata fields if present
            for key in ("level", "type", "source"):
                if key in ex:
                    meta[key] = ex[key]
            rows.append({
                "prompt": build_prompt_compmath(prob),
                "response": sol,
                "meta": meta
            })
        # If there were no rows because split missing, skip writing
        if rows:
            out_path = os.path.join(outdir, f"competition_math_{split}_alpaca.jsonl")
            write_jsonl(out_path, rows)
            print(f"[CompetitionMath] Wrote {len(rows)} rows to {out_path}")
    # If dataset exposes only a single split (e.g., "train"), ensure it's handled by the loop above.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="data", help="Output folder for JSONL files.")
    parser.add_argument("--gsm8k", action="store_true", help="Process GSM8K (openai/gsm8k).")
    parser.add_argument("--logiqa", action="store_true", help="Process LogiQA (lucasmccabe/logiqa).")
    parser.add_argument("--comp-math", action="store_true", help="Process Competition Math (qwedsacf/competition_math).")
    parser.add_argument("--all", action="store_true", help="Process all datasets.")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    if args.all or args.gsm8k:
        process_gsm8k(args.outdir)

    if args.all or args.logiqa:
        process_logiqa(args.outdir)

    if args.all or args.comp_math:
        process_competition_math(args.outdir)

    print("Done.")
    

if __name__ == "__main__":
    main()
