import argparse
import heapq
import json
from pathlib import Path
from typing import Any, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show the lowest reasoning_tokens samples from a JSONL dataset file."
        )
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the JSONL file to inspect.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help=(
            "How many samples to show. Positive for lowest tokens, negative for longest."
        ),
    )
    return parser.parse_args()


def collect_ranked_samples(path: Path, limit: int) -> List[Tuple[float, int, str]]:
    entries: List[Tuple[float, int, str]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "reasoning_tokens" not in record or "reasoning" not in record:
                continue
            try:
                token_value = float(record["reasoning_tokens"])
            except (TypeError, ValueError):
                continue

            reasoning_text = str(record["reasoning"])
            predict_text = str(record["predict"])
            entries.append((token_value, line_number, reasoning_text, predict_text))
    if limit > 0:
        return heapq.nsmallest(limit, entries, key=lambda item: item[0])
    limit = abs(limit)
    return heapq.nlargest(limit, entries, key=lambda item: item[0])


def main() -> None:
    args = parse_args()
    if not args.input_file.is_file():
        raise FileNotFoundError(f"File not found: {args.input_file}")
    if args.top == 0:
        raise ValueError("--top must not be zero.")

    samples = collect_ranked_samples(args.input_file, args.top)
    if not samples:
        print("No valid samples found.")
        return

    for idx, (token_value, line_number, reasoning_text, predict_text) in enumerate(samples, start=1):
        formatted_value = (
            f"{int(token_value)}" if token_value.is_integer() else f"{token_value}"
        )
        print(f"{idx}. line={line_number}, tokens={formatted_value}\n=== reasoning: {reasoning_text}\n=== predict: {predict_text}")


if __name__ == "__main__":
    main()
