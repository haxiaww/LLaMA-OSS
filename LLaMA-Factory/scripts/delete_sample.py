#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Expecting a JSON array of samples.")
    return data


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_no} does not contain a JSON object.")
            records.append(record)
    return records


def dump_json(path: Path, samples: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(samples), handle, ensure_ascii=False, indent=2)


def dump_jsonl(path: Path, samples: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False))
            handle.write("\n")


def prune_samples(
    samples: List[Dict[str, Any]],
    threshold: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    for sample in samples:
        sample_id = sample.get("id")
        grouped.setdefault(sample_id, []).append(sample)

    results: List[Dict[str, Any]] = []
    for sample_id, entries in grouped.items():
        overflow = max(0, len(entries) - threshold)
        if overflow == 0:
            results.extend(entries)
            continue

        # Drop items with larger generation_index first.
        sorted_entries = sorted(
            entries,
            key=lambda item: item.get("generation_index", 0),
            reverse=True,
        )
        kept = sorted_entries[overflow:]
        results.extend(kept)

    return results


def detect_format(path: Path) -> Tuple[str, bool]:
    if path.suffix.lower() == ".jsonl":
        return "jsonl", True
    if path.suffix.lower() == ".json":
        return "json", True
    return "unknown", False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove samples whose id appears more than the threshold number of times. "
            "When trimming, drop entries with the largest generation_index values first."
        ),
    )
    parser.add_argument("input_path", type=Path, help="Input JSON or JSONL file.")
    parser.add_argument(
        "--threshold",
        type=int,
        required=True,
        help="Maximum number of samples to keep for each id.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Destination file. Defaults to overwriting the input file.",
    )
    args = parser.parse_args()

    fmt, known = detect_format(args.input_path)
    if not known:
        raise ValueError("Input must have a .json or .jsonl extension.")

    samples = load_jsonl(args.input_path) if fmt == "jsonl" else load_json(args.input_path)
    pruned = prune_samples(samples, args.threshold)

    output_path = args.output_path or args.input_path
    fmt_out, known_out = detect_format(output_path)
    if not known_out:
        fmt_out = fmt  # fall back to input format if extension was omitted

    if fmt_out == "jsonl":
        dump_jsonl(output_path, pruned)
    else:
        dump_json(output_path, pruned)


if __name__ == "__main__":
    main()
