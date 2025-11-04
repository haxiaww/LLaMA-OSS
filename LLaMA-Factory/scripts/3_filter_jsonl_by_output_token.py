import json
import argparse
import os
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter JSONL files, keeping only records whose 'output_token' "
            "is greater than or equal to a threshold. Other fields are preserved."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=[
            "scripts/new_dataset/1_fbl/gptoss20b_gsm8k_train_results_high_fbl_tokens.jsonl",
            "scripts/new_dataset/1_fbl/gptoss20b_gsm8k_train_results_medium_fbl_tokens.jsonl",
            "scripts/new_dataset/1_fbl/gptoss20b_gsm8k_train_results_low_fbl_tokens.jsonl",
        ],
        help=(
            "Path(s) to input .jsonl file(s). Defaults to the three FBL files under repo. "
            "Pass multiple paths to combine before filtering."
        ),
    )
    parser.add_argument(
        "--output",
        default="scripts/new_dataset/2_fbt/filtered.jsonl",
        help="Path to output .jsonl file (default: training/.../2_fbt/filtered.jsonl)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help=(
            "Threshold for 'output_token'. Records with output_token < threshold are excluded."
        ),
    )
    return parser.parse_args()


def should_keep(record: Dict[str, Any], threshold: float) -> bool:
    if "output_token" not in record:
        return False
    try:
        value = float(record["output_token"])
    except (TypeError, ValueError):
        # Non-numeric -> drop
        return False
    return value >= threshold


def main() -> None:
    args = parse_args()

    # Process each input separately and write a corresponding output
    for input_path in args.input:
        kept = 0
        total = 0

        # Derive output path per input if multiple inputs
        if len(args.input) == 1 and args.output:
            out_path = args.output
        else:
            base = os.path.basename(input_path)
            name, _ = os.path.splitext(base)
            out_dir = os.path.dirname(args.output) or "2_fbt"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{name}_fbt_{str(int(args.threshold))}.jsonl")

        try:
            fin = open(input_path, "r", encoding="utf-8")
        except FileNotFoundError:
            # Skip missing files
            print(f"Input not found, skipping: {input_path}")
            continue

        with fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue

                if should_keep(obj, args.threshold):
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    kept += 1

        print(f"{input_path} -> {out_path}: processed {total}, kept {kept}, dropped {total - kept}")


if __name__ == "__main__":
    main()
