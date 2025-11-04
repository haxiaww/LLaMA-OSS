#!/usr/bin/env python3
"""Filter JSONL samples and emit template-based prompt/response records."""

import json
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILES = [
    BASE_DIR / "new_dataset" / "0_raw" / "gptoss20b_gsm8k_train_results_high.jsonl",
    BASE_DIR / "new_dataset" / "0_raw" / "gptoss20b_gsm8k_train_results_medium.jsonl",
    BASE_DIR / "new_dataset" / "0_raw" / "gptoss20b_gsm8k_train_results_low.jsonl",
]
DEFAULT_OUTPUT_DIR = BASE_DIR / "new_dataset" / "1_fbl"
DEFAULT_REJECTED_DIR = DEFAULT_OUTPUT_DIR / "fbl_rejected"
DEFAULT_OUTPUT_SUFFIX = "_fbl"
DEFAULT_REJECTED_SUFFIX = "_fbl_rejected"

# Drop special chat tokens from source fields before formatting the prompt.
TOKEN_PATTERN = re.compile(r"<\|[^>]*\|>")

REASONING_PROMPT_REMOVE_TOKENS = [
    "<|start|>user<|message|>",
    "<|end|><|start|>assistant",
]

LABEL_REMOVE_TOKENS = [
    "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>",
    "<|end|><|return|>",
]

# Template applied to each clean prompt; edit the constant to customize.
PROMPT_TEMPLATE = """Extract intermediate values from the reasoning process below.

RULES:
1. List each intermediate step result on a new line
2. Maintain the order they appear in the reasoning
3. Only include actual computed values
4. End with: #### <number of steps>

REASONING:
{reasoning}

OUTPUT FORMAT:
<value 1>
<value 2>
...
#### <count>
"""
PROMPT_PLACEHOLDER = "{reasoning}"


def extract_label(value: Any) -> Optional[str]:
    """Return trimmed answer text found after the first '####' token."""
    if not isinstance(value, str):
        return None
    parts = value.split("####", 1)
    if len(parts) < 2:
        return None
    return parts[1].strip().replace("<|end|><|return|>", "")


def extract_predict(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    parts = value.split("####", 1)
    if len(parts) < 2:
        return None
    return parts[1].strip()


def clean_text(raw_text: Any) -> str:
    """Strip special tokens and whitespace from source text."""
    if not isinstance(raw_text, str):
        return ""
    cleaned = TOKEN_PATTERN.sub("", raw_text)
    return cleaned.strip()


def clean_reasoning_prompt(raw_prompt: Any) -> str:
    """Remove specific chat tokens from the original prompt."""
    if not isinstance(raw_prompt, str):
        return ""
    cleaned = raw_prompt
    for token in REASONING_PROMPT_REMOVE_TOKENS:
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def clean_label(raw_label: Any) -> str:
    """Remove specific chat tokens from the label field."""
    if not isinstance(raw_label, str):
        return ""
    cleaned = raw_label
    for token in LABEL_REMOVE_TOKENS:
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def render_prompt(reasoning_text: str, template: str = PROMPT_TEMPLATE) -> str:
    """Inject the reasoning text into the configured template."""
    if PROMPT_PLACEHOLDER in template:
        return template.replace(PROMPT_PLACEHOLDER, reasoning_text)
    if template.endswith("\n"):
        return f"{template}{reasoning_text}".strip()
    return f"{template}\n\n{reasoning_text}".strip()


def build_prompt_response_record(
    sample: dict[str, Any],
    reasoning_prompt: str,
    step_prompt: str,
    label_text: str,
) -> dict[str, Any]:
    """Compose a record mirroring the original fields with clearer names."""
    return {
        "id": sample.get("id"),
        "generation_index": sample.get("generation_index"),
        "reasoning_prompt": reasoning_prompt,
        "gpt_reasoning": sample.get("predict"),
        "step_prompt": step_prompt,
        "label": label_text,
    }


def filter_samples_to_prompt_response(
    input_path: str, output_path: str, rejected_path: Optional[str] = None
) -> tuple[int, int]:
    """Write matching samples to output; optionally capture rejected ones."""
    kept = 0
    rejected = 0
    with open(input_path, "r", encoding="utf-8") as src, open(
        output_path, "w", encoding="utf-8"
    ) as dst, (
        open(rejected_path, "w", encoding="utf-8")
        if rejected_path
        else nullcontext()
    ) as rejected_file:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                if rejected_file is not None:
                    rejected_file.write(
                        raw_line if raw_line.endswith("\n") else raw_line + "\n"
                    )
                    rejected += 1
                continue

            predict_raw = sample.get("predict")
            label_raw = sample.get("label")
            predict = extract_predict(predict_raw)
            label = extract_label(label_raw)

            if (
                predict is not None
                and label is not None
                and predict == label
            ):
                original_prompt = clean_reasoning_prompt(sample.get("prompt"))
                reasoning_text = clean_text(sample.get("predict"))
                rendered_prompt = render_prompt(reasoning_text)
                prompt_response_record = build_prompt_response_record(
                    sample,
                    original_prompt,
                    rendered_prompt,
                    clean_label(sample.get("label")),
                )
                dst.write(json.dumps(prompt_response_record, ensure_ascii=False) + "\n")
                kept += 1
            elif rejected_file is not None:
                rejected_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                rejected += 1
    return kept, rejected


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for path if it does not already exist."""
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def derive_target_path(
    source: Path, target_dir: Optional[Path], suffix: str
) -> Path:
    """Build output path using suffix and optional target directory."""
    dest_dir = target_dir if target_dir is not None else source.parent
    filename = f"{source.stem}{suffix}{source.suffix or '.jsonl'}"
    return dest_dir / filename


def process_default_inputs(
    output_dir: Path, rejected_dir: Path
) -> None:
    """Run filtering over the default input file list."""
    for input_path in DEFAULT_INPUT_FILES:
        if not input_path.exists():
            print(f"[INFO] Missing input file: {input_path}; skipping.")
            continue
        output_path = derive_target_path(input_path, output_dir, DEFAULT_OUTPUT_SUFFIX)
        ensure_parent_dir(output_path)
        rejected_path = derive_target_path(
            input_path, rejected_dir, DEFAULT_REJECTED_SUFFIX
        )
        ensure_parent_dir(rejected_path)
        kept, rejected = filter_samples_to_prompt_response(
            str(input_path),
            str(output_path),
            str(rejected_path),
        )
        print(f"Kept {kept} samples from {input_path} -> {output_path}")
        print(f"Wrote {rejected} rejected samples to {rejected_path}")


def main() -> None:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    process_default_inputs(DEFAULT_OUTPUT_DIR, DEFAULT_REJECTED_DIR)


if __name__ == "__main__":
    main()
