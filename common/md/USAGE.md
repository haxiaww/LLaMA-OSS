# Usage Guide

Repo root = `LLaMA-OSS`. There is no top-level `src` Python package; workflows use **LLaMA-Factory scripts**, **MS-SWIFT CLI**, and optional **notebooks**.

## 1. Teacher inference (multi-mode CoT, vLLM)

Script: `LLaMA-Factory/scripts/0_gpt.py`  
Entry: `infer_vllm_pure` exposed via `fire.Fire` — pass **function kwargs as CLI flags**.

### Input JSONL (`0_gpt.py`)

Each line is one object. Required for generation:

- **`prompt`** — user text shown to the model.
- **`label`** — ground-truth string (often `\\boxed{...}`). Copied from **`response`** if `label` is missing.

Optional / passthrough fields are preserved on the sample dict as loaded.

### Run (from repo root)

```bash
cd LLaMA-Factory/scripts
python 0_gpt.py \
  --model_name_or_path <HF_OR_LOCAL_PATH_TO_GPT_OSS> \
  --dataset <path/to/prompts.jsonl> \
  --save_name <path/to/out.jsonl> \
  --mode low \
  --dataset_name gsm8k \
  --instruction "Respond concisely with minimal reasoning." \
  --generations_per_sample 5 \
  --max_new_tokens 2048 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.9
```

Repeat with **`--mode`** / **`--instruction`** for `medium`, `high` (match your paper setup).

### Useful flags (see docstring in file)

| Flag | Role |
|------|------|
| `dataset` | Absolute path to `.jsonl`, or dataset name under `dataset_dir` |
| `dataset_dir` | Default `data` (relative to CWD when resolving names) |
| `generations_per_sample` | vLLM `n` per prompt |
| `batch_size` | How many prompts to batch |
| `enable_thinking` | `True` → responses may include think-tags parsed by `extract_thinking_and_answer` (`think_pattern` in `0_gpt.py`) |
| `max_model_len` | Cap context; omit to use model default |
| `resume` | Same `--save_name`: completes missing generations, refills incomplete `\\boxed{}` |

Output lines include `prompt`, `response`, `mode`, `label`, `dataset`, `instruction`, `combined_index`, `_meta`, etc.

## 2. GRPO dataset conversion

`convert_data.ipynb` shows the pattern: read curated JSONL (with `prompt` / `label` or `response`), extract `\\boxed{...}` for **`label`**, write GRPO lines:

```json
{"query": "<same as prompt>", "label": "\\boxed{...}"}
```

`merged_grpo_data.jsonl` at repo root uses **`query`**, **`response`**, **`label`** — align with whatever **MS-SWIFT** expects for your reward (`grpo_accuracy` compares generations to label).

## 3. GRPO training (MS-SWIFT)

Example pattern: `train.sh` — **`swift rlhf`** with `--rlhf_type grpo`, `--dataset` pointing at your JSONL, LoRA on a student checkpoint.

Before running:

1. Edit **`train.sh`**: `CUDA_VISIBLE_DEVICES`, `--model`, `--dataset` path, `--output_dir`.
2. Run from repo root (with `swift` on `PATH` after `pip install -e ms-swift`):

```bash
bash train.sh
```

See [CONFIG.md](./CONFIG.md) for knobs.

## 4. Evaluation

See [EVALUATION.md](./EVALUATION.md) and root `eval.sh` (**lm-eval** + vLLM backend).
