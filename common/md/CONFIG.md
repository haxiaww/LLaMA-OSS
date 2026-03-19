# Configuration Guide

## Environment variables

| Variable | Purpose |
|----------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU selection (`scripts/train.sh`, `scripts/eval.sh`) |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Private / gated models on Hugging Face |
| `TOKENIZERS_PARALLELISM=false` | Avoid tokenizer warnings under multiprocessing |
| `VLLM_ENABLE_V1_MULTIPROCESSING=0` | Used in `scripts/eval.sh` for stable lm-eval + vLLM |

Optional:

- `USE_HF` — pass through MS-SWIFT / `swift rlhf` as needed; see MS-SWIFT docs.

## Teacher inference (`LLaMA-Factory/scripts/0_gpt.py`)

All settings are **CLI arguments** to `infer_vllm_pure` (Fire). There is no separate YAML for this script.

- **Model**: `--model_name_or_path` (local dir or Hub id).
- **Data**: `--dataset` (file path or name), `--dataset_dir`.
- **Mode label** (metadata only in output): `--mode`, `--dataset_name`, `--instruction`.
- **Sampling**: `--temperature`, `--top_p`, `--top_k`, `--max_new_tokens`, `--generations_per_sample`, `--seed`.
- **vLLM**: `--tensor_parallel_size`, `--gpu_memory_utilization`, `--max_model_len`, `--dtype`, `--trust_remote_code`.

Working directory: run from `LLaMA-Factory/scripts` or pass **absolute** `--dataset` / `--save_name` so relative paths resolve correctly.

## GRPO training (`scripts/train.sh` → `swift rlhf`)

Edit defaults in `scripts/train.sh` or export `MODEL`, `DATASET`, `OUTPUT_DIR` before running:

| CLI flag | Typical meaning here |
|----------|----------------------|
| `--model` | Student base or SFT adapter checkpoint (Hub or path) |
| `--model_type` | e.g. `llama3_2` |
| `--dataset` | Comma-separated JSONL paths if needed |
| `--rlhf_type grpo` | GRPO |
| `--reward_funcs` | e.g. `grpo_accuracy` |
| `--train_type lora` | LoRA |
| `--max_steps`, `--per_device_train_batch_size`, `--gradient_accumulation_steps` | Training length / effective batch |
| `--max_length` | Sequence length |
| `--output_dir` | Checkpoints |
| `--num_generations` / sampling flags | On-policy rollouts (see MS-SWIFT GRPO docs) |

`DATASET` / `OUTPUT_DIR` default under repo root; override with absolute paths if your data lives elsewhere.

## Datasets layout

- Alpaca-style / train JSONL often lives under **`LLaMA-Factory/data/`** (see defaults in `scripts/convert_data.sh`).
- Repo root may hold merged GRPO files (e.g. `merged_grpo_data.jsonl`, `train_grpo.jsonl`) — **field names differ** by pipeline stage; see [USAGE.md](./USAGE.md).

## Dependencies

Root `requirements.txt` lists pandas/numpy/tqdm/jupyter/etc. **LLaMA-Factory** and **ms-swift** are installed separately (`pip install -e .` in each subfolder). Optional: `flash-attn` — install manually; it is not a normal single-line `requirements.txt` entry.
