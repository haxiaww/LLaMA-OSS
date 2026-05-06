#!/usr/bin/env python3
"""
GRPO / JSONL prep (same behavior as the old convert_data notebook).
Default paths assume repo layout: LLaMA-Factory/data/...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def _repo_root() -> Path:
    env = os.environ.get("REPO_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


def cmd_repo_root(_: argparse.Namespace) -> None:
    print(_repo_root())


def _extract_boxed_simple(text: str) -> str:
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    return f"\\boxed{{{m.group(1)}}}" if m else ""


def cmd_grpo_high(ns: argparse.Namespace) -> None:
    input_file = Path(ns.input_file)
    output_file = Path(ns.output_file)
    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line)
            label = _extract_boxed_simple(data.get("label", ""))
            row = {"query": data.get("prompt", ""), "label": label}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {output_file}")


def cmd_compmath(ns: argparse.Namespace) -> None:
    input_file = Path(ns.input_file)
    output_file = Path(ns.output_file)
    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line)
            label = _extract_boxed_simple(data.get("response", ""))
            row = {
                "query": data.get("prompt", ""),
                "response": data.get("response", ""),
                "label": label,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {output_file}")


def cmd_merge(ns: argparse.Namespace) -> None:
    paths = [Path(p) for p in ns.inputs]
    output_file = Path(ns.output_file)
    with open(output_file, "w", encoding="utf-8") as outfile:
        for path in paths:
            with open(path, "r", encoding="utf-8") as infile:
                for line in infile:
                    if line.strip():
                        outfile.write(line.strip() + "\n")
    print(f"Merged -> {output_file}")


def extract_answer_from_response(response: str, dataset: str):
    if not response:
        return None
    if dataset.lower() == "logiqa":
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if think_match:
            thinking = think_match.group(1)
            answer_match = re.search(
                r"(?:answer|option|choice)\s*(?:is|:|=)?\s*([A-D])\b", thinking, re.IGNORECASE
            )
            if answer_match:
                return answer_match.group(1).upper()
            letters = re.findall(r"\b([A-D])\b", thinking)
            if letters:
                return letters[-1].upper()
    answer_match = re.search(r"####\s*(.+?)(?:\n|</think>|$)", response)
    if answer_match:
        return answer_match.group(1).strip()
    boxed_match = re.search(r"\\boxed\{", response)
    if boxed_match:
        text = response[boxed_match.end() :]
        brace_count = 1
        end_pos = 0
        for i, char in enumerate(text):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i
                    break
        if end_pos > 0:
            return text[:end_pos].strip()
    return None


def convert_to_grpo_format(input_path: Path, output_path: Path) -> None:
    stats = {"total": 0, "answer_extracted": 0, "no_answer": 0, "by_dataset": {}}
    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            item = json.loads(line)
            stats["total"] += 1
            prompt = item.get("prompt", "")
            instruction = item.get("instruction", "Respond concisely with minimal reasoning.")
            response = item.get("response", "")
            mode = item.get("mode", "low")
            dataset = item.get("dataset", "unknown")
            bd = stats["by_dataset"].setdefault(dataset, {"total": 0, "success": 0})
            bd["total"] += 1
            format_instruction = (
                "provide your reasoning in <think> tags, then put your final answer in \\boxed{}"
            )
            query = f"{instruction}\n\n{format_instruction}\n\n{prompt}"
            extracted = extract_answer_from_response(response, dataset)
            if extracted:
                stats["answer_extracted"] += 1
                bd["success"] += 1
                if stats["answer_extracted"] <= 5:
                    print(f"  extracted: {extracted!r}")
                ground_truth = f"\\boxed{{{extracted}}}"
            else:
                stats["no_answer"] += 1
                ground_truth = "\\boxed{unknown}"
            row = {"query": query, "mode": mode, "answer": ground_truth}
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
    t = stats["total"]
    print(f"done: {t} lines, extracted {stats['answer_extracted']}, missing {stats['no_answer']}")
    for ds, c in stats["by_dataset"].items():
        rate = 100 * c["success"] / c["total"] if c["total"] else 0
        print(f"  {ds}: {c['success']}/{c['total']} ({rate:.1f}%)")


def cmd_combined_grpo(ns: argparse.Namespace) -> None:
    combined_in = Path(ns.input_file)
    train_out = Path(ns.output_file)
    convert_to_grpo_format(combined_in, train_out)
    print("samples:")
    with open(train_out, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(f"  {i + 1}. {json.loads(line)['answer']}")


def cmd_analyze_thinking(ns: argparse.Namespace) -> None:
    import numpy as np

    files = [Path(p) for p in ns.inputs]
    results = {}
    for file_path in files:
        fp = str(file_path)
        print(f"\n{'=' * 80}\n{fp}\n{'=' * 80}")
        mode_lengths = defaultdict(list)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    mode = item.get("mode", "unknown").lower()
                    response = item.get("response", "")
                    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
                    if match:
                        thinking = match.group(1).strip()
                        mode_lengths[mode].append(len(thinking) // 4)
        except FileNotFoundError:
            print(f"missing: {fp}")
            continue
        file_results = {}
        for mode in ("low", "medium", "high"):
            if mode not in mode_lengths:
                continue
            lengths = mode_lengths[mode]
            file_results[mode] = {
                "count": len(lengths),
                "mean": float(np.mean(lengths)),
                "std": float(np.std(lengths)),
                "min": min(lengths),
                "max": max(lengths),
                "median": float(np.median(lengths)),
            }
            print(
                f"{mode}: n={len(lengths)} mean={file_results[mode]['mean']:.1f} "
                f"std={file_results[mode]['std']:.1f}"
            )
        results[fp] = file_results
    print("\nreward snippet (mode_params):")
    for file_path, modes in results.items():
        name = Path(file_path).name
        print(f"# {name}")
        print("self.mode_params = {")
        for mode in ("low", "medium", "high"):
            if mode in modes:
                m, s = int(modes[mode]["mean"]), int(modes[mode]["std"])
                print(f"    '{mode}': {{'mean': {m}, 'std': {s}}},")
        print("}")


def main() -> None:
    root = _repo_root()
    p = argparse.ArgumentParser(description="GRPO / JSONL conversion (defaults under repo root).")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("repo-root", help="Print detected repo root").set_defaults(fn=cmd_repo_root)

    g = sub.add_parser("grpo-high", help="grpo_high.jsonl from LLaMA-Factory/data/final/grpo_high.jsonl")
    g.add_argument(
        "--input-file",
        type=Path,
        default=root / "LLaMA-Factory/data/final/grpo_high.jsonl",
    )
    g.add_argument("--output-file", type=Path, default=root / "grpo_high.jsonl")
    g.set_defaults(fn=cmd_grpo_high)

    c = sub.add_parser("compmath", help="compmath_grpo.jsonl from competition_math alpaca JSONL")
    c.add_argument(
        "--input-file",
        type=Path,
        default=root / "LLaMA-Factory/data/competition_math_origin_train_alpaca.jsonl",
    )
    c.add_argument("--output-file", type=Path, default=root / "compmath_grpo.jsonl")
    c.set_defaults(fn=cmd_compmath)

    m = sub.add_parser("merge", help="Concatenate JSONL files (default: compmath + gsm8k -> merged_grpo_data)")
    m.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[root / "compmath_grpo.jsonl", root / "gsm8k_grpo.jsonl"],
    )
    m.add_argument("--output-file", type=Path, default=root / "merged_grpo_data.jsonl")
    m.set_defaults(fn=cmd_merge)

    cg = sub.add_parser("combined-grpo", help="combined_grpo_train.jsonl -> train_grpo.jsonl")
    cg.add_argument(
        "--input-file",
        type=Path,
        default=root / "LLaMA-Factory/data/combined_grpo_train.jsonl",
    )
    cg.add_argument("--output-file", type=Path, default=root / "train_grpo.jsonl")
    cg.set_defaults(fn=cmd_combined_grpo)

    a = sub.add_parser("analyze-thinking", help="Thinking-length stats on combined_sft_train_{high,low,medium}")
    a.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            root / "LLaMA-Factory/data/combined_sft_train_high.jsonl",
            root / "LLaMA-Factory/data/combined_sft_train_low.jsonl",
            root / "LLaMA-Factory/data/combined_sft_train_medium.jsonl",
        ],
    )
    a.set_defaults(fn=cmd_analyze_thinking)

    ns = p.parse_args()
    ns.fn(ns)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
