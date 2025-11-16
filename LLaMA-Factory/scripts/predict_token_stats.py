import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from transformers import AutoTokenizer

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "llama3": {
        "label": "Llama-3.2-3B-Instruct",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "trust_remote_code": False,
    },
    "qwen2.5": {
        "label": "Qwen2.5-3B-Instruct",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "trust_remote_code": True,
    },
    "gpt": {
        "label": "OpenAI GPT-OSS-20B",
        "model_id": "openai/gpt-oss-20b",
        "trust_remote_code": False,
    },
}

FILENAME_KEYWORDS: Dict[str, Sequence[str]] = {
    "llama3": ("llama",),
    "qwen2.5": ("qwen",),
    "gpt": ("gpt",),
}

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "predict_token_stats"


def summarize(lengths: Sequence[int]) -> Optional[Dict[str, float]]:
    if not lengths:
        return None
    std_dev = statistics.pstdev(lengths)
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": statistics.fmean(lengths),
        "median": statistics.median(lengths),
        "std": std_dev,
    }


def render_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No statistics to display.")
        return

    if tabulate:
        print(tabulate(rows, headers="keys", tablefmt="github"))
        return

    headers = rows[0].keys()
    col_widths = {
        header: max(len(header), *(len(str(row[header])) for row in rows))
        for header in headers
    }
    header_line = " | ".join(f"{header:{col_widths[header]}}" for header in headers)
    separator = "-+-".join("-" * col_widths[header] for header in headers)
    print(header_line)
    print(separator)
    for row in rows:
        print(" | ".join(f"{str(row[header]):{col_widths[header]}}" for header in headers))


def normalize_predict_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return " ".join(normalize_predict_value(item) for item in value)
    return str(value)


def iter_jsonl_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc


def compute_lengths(
    path: Path,
    tokenizer: AutoTokenizer,
    field: str,
    add_special_tokens: bool,
) -> List[int]:
    lengths: List[int] = []
    for record in iter_jsonl_records(path):
        if field not in record or record[field] is None:
            continue
        raw_text = normalize_predict_value(record[field])
        if not raw_text:
            continue
        tokens = tokenizer(raw_text, add_special_tokens=add_special_tokens)
        lengths.append(len(tokens["input_ids"]))
    return lengths


def save_histogram(lengths: Sequence[int], output_dir: Path) -> Optional[Path]:
    if not lengths:
        print("No predict tokens were found; skipping histogram generation.")
        return None
    if plt is None:
        print("matplotlib is not installed; skipping histogram generation.")
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins="auto", alpha=0.85, edgecolor="black")
    ax.set_title("Predict Token Distribution")
    ax.set_xlabel("Token count")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / "predict_token_distribution.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def collect_jsonl_paths(input_dir: Path) -> List[Path]:
    if not input_dir.is_dir():
        raise ValueError(f"Input path {input_dir} is not a directory")
    return sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jsonl"
    )


def detect_model_key_from_filename(path: Path) -> str:
    lower_name = path.name.lower()
    for key, keywords in FILENAME_KEYWORDS.items():
        if any(keyword in lower_name for keyword in keywords):
            return key
    raise ValueError(f"Unable to infer tokenizer preset from filename {path}")


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize a folder of JSONL files and summarize the 'predict' token lengths."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        required=True,
        help="Directory that contains .jsonl files with a common schema.",
    )
    parser.add_argument(
        "--field",
        "-f",
        type=str,
        default="predict",
        help="Field name that contains the text to tokenize (default: predict).",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_PRESETS.keys()),
        default=None,
        help="Force a preset tokenizer for all files instead of inferring from filenames.",
    )
    parser.add_argument(
        "--tokenizer-id",
        type=str,
        help="Explicit Hugging Face tokenizer ID (overrides preset).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Set to true when the tokenizer requires trusting remote code (defaults from preset).",
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Include the tokenizer's special tokens when counting.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional cache directory for the tokenizer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where summary artifacts (histogram) are written.",
    )
    return parser.parse_args()


def main():
    args = build_args()
    jsonl_paths = collect_jsonl_paths(args.input_dir)
    if not jsonl_paths:
        raise SystemExit(f"No .jsonl files found in {args.input_dir}")

    if args.tokenizer_id and args.model is None:
        raise SystemExit("--tokenizer-id requires --model when inferring tokenizers per file.")

    cache_dir = str(args.cache_dir) if args.cache_dir else None
    tokenizer_cache: Dict[str, AutoTokenizer] = {}

    def get_tokenizer(model_key: str) -> AutoTokenizer:
        if model_key in tokenizer_cache:
            return tokenizer_cache[model_key]
        preset = MODEL_PRESETS[model_key]
        tokenizer_id = args.tokenizer_id or preset["model_id"]
        trust_remote_code = (
            preset["trust_remote_code"]
            if args.trust_remote_code is None
            else args.trust_remote_code
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        tokenizer_cache[model_key] = tokenizer
        return tokenizer

    per_file_lengths: List[Tuple[Path, str, List[int]]] = []
    for path in jsonl_paths:
        try:
            model_key = args.model or detect_model_key_from_filename(path)
        except ValueError as exc:
            print(f"Skipping {path.name}: {exc}")
            continue
        tokenizer = get_tokenizer(model_key)
        lengths = compute_lengths(
            path,
            tokenizer,
            args.field,
            add_special_tokens=args.add_special_tokens,
        )
        if lengths:
            per_file_lengths.append((path, model_key, lengths))
        else:
            preset_label = MODEL_PRESETS[model_key]["label"]
            print(f"Warning: no valid entries found in {path} (model {preset_label})")

    if not per_file_lengths:
        raise SystemExit("No tokenized entries found for the requested field.")

    combined_lengths: List[int] = []
    table_rows: List[Dict[str, Any]] = []
    for path, model_key, lengths in per_file_lengths:
        combined_lengths.extend(lengths)
        summary = summarize(lengths)
        if not summary:
            continue
        table_rows.append(
            {
                "File": path.name,
                "Model": MODEL_PRESETS[model_key]["label"],
                "Count": summary["count"],
                "Min": summary["min"],
                "Max": summary["max"],
                "Mean": round(summary["mean"], 2),
                "Median": summary["median"],
                "Std": round(summary["std"], 2),
            }
        )

    aggregate_summary = summarize(combined_lengths)
    if aggregate_summary:
        table_rows.append(
            {
                "File": "All files",
                "Model": "Combined",
                "Count": aggregate_summary["count"],
                "Min": aggregate_summary["min"],
                "Max": aggregate_summary["max"],
                "Mean": round(aggregate_summary["mean"], 2),
                "Median": aggregate_summary["median"],
                "Std": round(aggregate_summary["std"], 2),
            }
        )

    print(
        f"Field: {args.field}, processed files: {len(per_file_lengths)} / {len(jsonl_paths)}"
    )
    render_table(table_rows)

    hist_path = save_histogram(combined_lengths, args.output_dir)
    if hist_path:
        print(f"Saved combined histogram to {hist_path}")


if __name__ == "__main__":
    main()
