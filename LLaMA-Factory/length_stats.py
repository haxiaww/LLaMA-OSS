import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from tabulate import tabulate
except ImportError:  # pragma: no cover - optional dependency
    tabulate = None

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


# Default input files. Adjust as needed.
DEFAULT_FILENAMES = [
    "gptoss20b_logiqa_train_results_high.jsonl",
    "gptoss20b_logiqa_train_results_medium.jsonl",
    "gptoss20b_logiqa_train_results_low.jsonl",
]

HIST_OUTPUT_DIR = Path(__file__).parent / "length_histograms"

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


def render_table(table_rows: List[Dict[str, object]]) -> None:
    if not table_rows:
        print("No data found for the requested fields.")
        return

    if tabulate:
        print(tabulate(table_rows, headers="keys", tablefmt="github"))
        return

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


def compute_stats(path: Path, fields: Iterable[str]) -> Dict[str, Dict[str, object]]:
    stats_data = {
        field: {
            "lengths": [],
            "min": None,  # (length, line_number)
            "max": None,  # (length, line_number)
        }
        for field in fields
    }

    with path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            for field in fields:
                if field not in record or record[field] is None:
                    continue
                # Read numeric value directly from field (expects 'output_tokens')
                try:
                    length = int(record[field])
                except Exception:
                    # Skip if cannot coerce to int
                    continue
                data = stats_data[field]
                data["lengths"].append(length)

                current_min = data["min"]
                if current_min is None or length < current_min[0]:
                    data["min"] = (length, line_number)

                current_max = data["max"]
                if current_max is None or length > current_max[0]:
                    data["max"] = (length, line_number)

    results: Dict[str, Dict[str, object]] = {}
    for field, data in stats_data.items():
        summary = summarize(data["lengths"])
        if summary is None:
            continue

        min_length, min_line = data["min"]
        max_length, max_line = data["max"]
        results[field] = {
            "table": {
                "Field": field,
                "Count": summary["count"],
                "Min Length": min_length,
                "Min Line": min_line,
                "Max Length": max_length,
                "Max Line": max_line,
                "Mean": round(summary["mean"], 2),
                "Median": summary["median"],
            },
            "lengths": data["lengths"],
        }
    return results


def save_predict_histograms(entries: List[Tuple[Path, List[int]]]) -> Optional[Path]:
    filtered_entries = [(path, lengths) for path, lengths in entries if lengths]
    if not filtered_entries:
        return None

    if plt is None:
        print("matplotlib is not installed; skipping histogram generation.")
        return None

    HIST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = HIST_OUTPUT_DIR / "predict_length_histograms.png"

    num_plots = len(filtered_entries)
    fig, axes = plt.subplots(
        1,
        num_plots,
        figsize=(6 * num_plots, 4.5),
        squeeze=False,
    )
    axes_flat = axes[0]
    for ax, (path, lengths) in zip(axes_flat, filtered_entries):
        ax.hist(lengths, bins="auto", color="#3A7BD5", alpha=0.85, edgecolor="black")
        ax.set_title(path.name)
        ax.set_xlabel("Length (tokens)")
        ax.set_ylabel("Frequency")
        ax.set_ylim(0, 2500)
        ax.set_xlim(0, 450)

    # Hide any unused axes when fewer plots than allocated
    for ax in axes_flat[len(filtered_entries):]:
        ax.set_visible(False)

    fig.suptitle("Predict Length Distribution")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _friendly_label(path: Path) -> str:
    name = path.name.lower()
    if "high" in name:
        return "high"
    if "medium" in name:
        return "medium"
    if "low" in name:
        return "low"
    return path.stem


def save_comparative_plots(entries: List[Tuple[Path, List[int]]]) -> List[Path]:
    if plt is None:
        print("matplotlib is not installed; skipping comparative plots.")
        return []

    entries = [(p, lst) for p, lst in entries if lst]
    if not entries:
        return []

    HIST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []

    # 1) Overlaid histograms (normalized)
    fig, ax = plt.subplots(figsize=(8, 5))
    for path, lengths in entries:
        ax.hist(lengths, bins="auto", alpha=0.4, density=True, label=_friendly_label(path))
    ax.set_title("Output Token Distribution (normalized)")
    ax.set_xlabel("Length (tokens)")
    ax.set_ylabel("Density")
    ax.legend()
    out1 = HIST_OUTPUT_DIR / "compare_hist_overlaid.png"
    fig.tight_layout()
    fig.savefig(out1, dpi=200)
    plt.close(fig)
    outputs.append(out1)

    # 2) Empirical CDF comparison
    import numpy as np
    fig, ax = plt.subplots(figsize=(8, 5))
    for path, lengths in entries:
        arr = np.sort(np.array(lengths))
        y = np.arange(1, len(arr) + 1) / len(arr)
        ax.plot(arr, y, label=_friendly_label(path))
    ax.set_title("Empirical CDF of Output Tokens")
    ax.set_xlabel("Length (tokens)")
    ax.set_ylabel("CDF")
    ax.legend()
    out2 = HIST_OUTPUT_DIR / "compare_cdf.png"
    fig.tight_layout()
    fig.savefig(out2, dpi=200)
    plt.close(fig)
    outputs.append(out2)

    # 3) Box/Violin plots for summary
    labels = [_friendly_label(p) for p, _ in entries]
    data = [lst for _, lst in entries]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Matplotlib 3.9+: use tick_labels to avoid deprecation warning
    axes[0].boxplot(data, tick_labels=labels, showfliers=False)
    axes[0].set_title("Boxplot of Output Tokens")
    axes[0].set_ylabel("Length (tokens)")
    parts = axes[1].violinplot(data, showmeans=True, showextrema=True, showmedians=True)
    axes[1].set_title("Violin Plot of Output Tokens")
    axes[1].set_xticks(range(1, len(labels) + 1))
    axes[1].set_xticklabels(labels, rotation=15)
    out3 = HIST_OUTPUT_DIR / "compare_box_violin.png"
    fig.tight_layout()
    fig.savefig(out3, dpi=200)
    plt.close(fig)
    outputs.append(out3)

    return outputs


def dedupe_representative_fields(
    rows: List[Dict[str, object]], representative_fields: Iterable[str]
) -> List[Dict[str, object]]:
    seen = set()
    filtered_rows = []
    rep_fields = set(representative_fields)
    for row in rows:
        field = row.get("Field")
        if field in rep_fields:
            if field in seen:
                continue
            seen.add(field)
        filtered_rows.append(row)
    return filtered_rows


def main():
    base_dir = Path(__file__).parent
    paths = [base_dir / filename for filename in DEFAULT_FILENAMES]

    processed_results: List[Tuple[Path, Dict[str, Dict[str, object]]]] = []
    for path in paths:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        # Only care about the numeric field 'output_tokens' in inputs
        results = compute_stats(path, fields=["output_tokens"])
        if not results:
            print(f"No data found for the requested fields in {path}")
            continue

        processed_results.append((path, results))

    combined_rows = []
    predict_entries: List[Tuple[Path, List[int]]] = []
    for path, results in processed_results:
        field = "output_tokens"
        field_data = results.get(field)
        if not field_data:
            continue
        row = {"File": path.name}
        row.update(field_data["table"])
        combined_rows.append(row)
        predict_entries.append((path, field_data["lengths"]))

    print("\nCombined statistics (by output_tokens):")
    render_table(combined_rows)

    hist_path = save_predict_histograms(predict_entries)
    if hist_path:
        print(f"\nPredict length histograms saved to: {hist_path}")
    comp_paths = save_comparative_plots(predict_entries)
    if comp_paths:
        print("Comparative plots saved:")
        for p in comp_paths:
            print(f" - {p}")


if __name__ == "__main__":
    main()
