import argparse
import json
import statistics
from pathlib import Path

try:
    from tabulate import tabulate
except ImportError:  # pragma: no cover - optional dependency
    tabulate = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute basic string-length statistics for fields in a JSONL file."
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="Path to the JSONL file to analyze.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["prompt", "predict", "label"],
        help="JSON keys to measure. Defaults to prompt predict label.",
    )
    return parser


def summarize(lengths):
    if not lengths:
        return None

    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": statistics.fmean(lengths),
        "median": statistics.median(lengths),
    }


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.path.exists():
        parser.error(f"File not found: {args.path}")

    stats_data = {
        field: {
            "lengths": [],
            "min": None,  # (length, line_number)
            "max": None,  # (length, line_number)
        }
        for field in args.fields
    }

    with args.path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            for field in args.fields:
                if field not in record or record[field] is None:
                    continue
                value = record[field]
                if not isinstance(value, str):
                    value = str(value)
                length = len(value)
                data = stats_data[field]
                data["lengths"].append(length)

                current_min = data["min"]
                if current_min is None or length < current_min[0]:
                    data["min"] = (length, line_number)

                current_max = data["max"]
                if current_max is None or length > current_max[0]:
                    data["max"] = (length, line_number)

    for field in args.fields:
        data = stats_data[field]
        summary = summarize(data["lengths"])
        if summary is None:
            continue

        min_length, min_line = data["min"]
        max_length, max_line = data["max"]
        stats_data[field] = {
            "Field": field,
            "Count": summary["count"],
            "Min Length": min_length,
            "Min Line": min_line,
            "Max Length": max_length,
            "Max Line": max_line,
            "Mean": round(summary["mean"], 2),
            "Median": summary["median"],
        }

    table_rows = [
        stats_data[field]
        for field in args.fields
        if isinstance(stats_data[field], dict) and "Field" in stats_data[field]
    ]
    if not table_rows:
        print("No data found for the requested fields.")
        return

    if tabulate:
        print(tabulate(table_rows, headers="keys", tablefmt="github"))
    else:
        # Basic fallback table formatting if tabulate is unavailable.
        headers = table_rows[0].keys()
        col_widths = {
            header: max(len(header), *(len(str(row[header])) for row in table_rows))
            for header in headers
        }
        header_line = " | ".join(f"{header:{col_widths[header]}}" for header in headers)
        separator = "-+-".join("-" * col_widths[header] for header in headers)
        print(header_line)
        print(separator)
        for row in table_rows:
            print(" | ".join(f"{str(row[header]):{col_widths[header]}}" for header in headers))


if __name__ == "__main__":
    main()
