# Configuration Guide

## Environment variables

| Variable | Purpose |
|----------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU selection (`scripts/train.sh` default `2`, `scripts/eval.sh` positional or export) |
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

Edit `scripts/train.sh` or export env before `bash scripts/train.sh`:

- **`MODEL`**, **`MODEL_TYPE`**, **`DATASET`**, **`OUTPUT_DIR`**
- **`CUDA_VISIBLE_DEVICES`** (script default `2`)
- **`USE_HF=0`** to omit `--use_hf`

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

- Alpaca-style / train JSONL often lives under **`LLaMA-Factory/data/`** (see defaults in `scripts/convert_data.py`).
- Repo root may hold merged GRPO files (e.g. `merged_grpo_data.jsonl`, `train_grpo.jsonl`) — **field names differ** by pipeline stage; see [USAGE.md](./USAGE.md).

## Model and training overview

- **Base model**: Llama 3.2 3B (e.g. `meta-llama/Llama-3.2-3B-Instruct`). SFT/GRPO đều dùng LoRA.
- **SFT**: LLaMA-Factory (`stage: sft`, `finetuning_type: lora`), config YAML trong `LLaMA-Factory/examples/train_lora/`.
- **GRPO**: MS-SWIFT (`swift rlhf --rlhf_type grpo --train_type lora`), script `scripts/train.sh`.

---

## Config từng bước

### 1. SFT (LLaMA-Factory, LoRA)

Chạy từ thư mục LLaMA-Factory: `llamafactory-cli train <yaml>`.

**Dataset (tên trong LLaMA-Factory):**

| Tên dataset | Ý nghĩa | Nguồn data (DVC) |
|-------------|---------|-------------------|
| `sft_low` | CoT ngắn (low mode) | `data/0_raw/gsm8k/gsm8k_train_low.jsonl` + compmath low (sau pipeline) |
| `sft_medium` | CoT trung bình (med) | `gsm8k_train_med.jsonl` + compmath_med |
| `sft_high` | CoT dài (high mode) | `gsm8k_train_high.jsonl` + compmath_high |
| origin | Data gốc / chưa tách mode | Tùy cách đăng ký trong `dataset_info` (data có thể từ merged hoặc alpaca gốc) |

**Config SFT chung (Llama, LoRA) — từ `llama_1mode.yaml` / `llama_3mode.yaml`:**

| Tham số | Giá trị | Ghi chú |
|---------|---------|--------|
| `model_name_or_path` | `meta-llama/Llama-3.2-3B-Instruct` | Base model |
| `stage` | `sft` | |
| `finetuning_type` | `lora` | |
| `lora_rank` | `32` | Rank LoRA (repo dùng 32) |
| `lora_alpha` | `64` | Scaling alpha |
| `lora_target` | `all` | |
| `lora_dropout` | `0.05` | |
| `cutoff_len` | `3072` | Max sequence length (prompt + response) |
| `learning_rate` | `2.0e-4` | |
| `per_device_train_batch_size` | `16` | |
| `gradient_accumulation_steps` | `2` | Effective batch = 32 |
| `num_train_epochs` | `1.0` (1 mode) / có thể 0.3 cho phase1 | |
| `lr_scheduler_type` | `cosine` | |
| `warmup_ratio` | `0.1` | |
| `bf16` | `true` | |
| `template` | `llama3` | |

**Config SFT theo dataset (dataset field trong YAML):**

| Config | Dataset (train) | Eval | File tham khảo |
|--------|------------------|------|------------------|
| **Origin** | 1 dataset gốc (cần đăng ký trong dataset_info) | tương ứng | Tự tạo YAML hoặc dùng 1 trong các mode |
| **Low only** | `sft_low` | `sft_grpo_val_low,sft_grpo_val_medium,sft_grpo_val_high` | Tạo từ `llama_1mode.yaml`, đổi `dataset: sft_low` |
| **Med only** | `sft_medium` | idem | `llama_1mode.yaml` (đã dùng sft_medium) |
| **High only** | `sft_high` | idem | Tạo từ `llama_1mode.yaml`, đổi `dataset: sft_high` |
| **Low + Med + High** | `sft_low,sft_medium,sft_high` | `sft_grpo_val_low,sft_grpo_val_medium,sft_grpo_val_high` | `llama_3mode.yaml` |
| **Curriculum (phase1)** | `sft_low,sft_medium` mix 0.7/0.3 | `sft_grpo_val_low,sft_grpo_val_medium` | `llama_phase1.yaml` |
| **Curriculum (phase2/3)** | `sft_low,sft_medium,sft_high` | đủ 3 val | `llama_phase2.yaml`, `llama_phase3.yaml` |

`output_dir` / `run_name` đổi theo run (vd: `saves/llama_3b_sft_low`, `saves/llama_3b_sft_origin`).

---

### 2. GRPO (MS-SWIFT, LoRA)

Chạy từ repo root: `bash scripts/train.sh` (hoặc set env rồi chạy).

**Dataset GRPO:** JSONL có `query` và `label` (đáp án `\boxed{...}`). Nguồn tạo: `scripts/convert_data.py` (merge, combined-grpo, …) → `merged_grpo_data.jsonl` hoặc `train_grpo.jsonl`.

**Config GRPO trong `scripts/train.sh` (và override bằng env):**

| Tham số | Giá trị hiện tại | Mô tả |
|---------|-------------------|--------|
| `MODEL` | `KoiiVN/final_llama_3b_sft_origin` | Base hoặc checkpoint SFT (Hub/path) |
| `MODEL_TYPE` | `llama3_2` | Kiểu model |
| `DATASET` | `merged_grpo_data.jsonl` | Path JSONL GRPO |
| `OUTPUT_DIR` | `outputs/llama_origin_grpo` | Thư mục checkpoint |
| `--max_length` | `3072` | Max sequence length (prompt + completion) |
| `--per_device_train_batch_size` | `8` | |
| `--gradient_accumulation_steps` | `2` | |
| `--max_steps` | `300` | |
| `--train_type` | `lora` | LoRA |
| `--loss_type` | `dapo` | Biến thể GRPO |
| `--reward_funcs` | `grpo_accuracy` | Reward chính (math accuracy + repetition bonus) |
| `--reward_weights` | `1` | Trọng số cho từng reward (1 phần tử vì 1 reward) |
| `--num_generations` | `8` | Số rollout mỗi prompt |
| `--temperature` | `1.0` | Sampling |
| `--warmup_ratio` | `0.05` | |
| `--save_steps` | `100` | |
| `--save_total_limit` | `4` | |
| `--bf16` | `true` | |
| `--gradient_checkpointing` | `true` | |

**Tham số GRPO không có trong script (dùng default MS-SWIFT):**

| Tham số | Default (LoRA) | Ghi chú |
|---------|----------------|--------|
| `learning_rate` | `1e-4` | Có thể thêm `--learning_rate 1e-4` vào `train.sh` nếu muốn chỉ rõ |
| `lora_rank` | `8` | Megatron/SWIFT default 8; muốn giống SFT (32) thì thêm `--lora_rank 32` |
| `lora_alpha` | `32` | Có thể thêm `--lora_alpha 32` (hoặc 64) |

**Maximum generation length (GRPO):** Trong MS-SWIFT GRPO, độ dài phần “generate” được giới hạn bởi `max_length` (toàn sequence). Không tách riêng “max_new_tokens” trong `train.sh`; cần thì xem tham số `max_completion_length` / generation trong docs MS-SWIFT và thêm vào lệnh nếu có.

**Reward functions (MS-SWIFT, `swift/plugin/orm.py`):**

| Tên (`--reward_funcs`) | Mô tả |
|------------------------|--------|
| `grpo_accuracy` | Accuracy math (so `\boxed{}` với label) + repetition penalty → bonus 0~0.5 cho câu đúng, ngắn gọn. Dùng chính cho math. |
| `accuracy` | Chỉ math accuracy (MathAccuracy). |
| `format` | Khớp format `<think>...</think> <answer>...</answer>`. |
| `react_format` | Khớp format ReAct (Action / Action Input). |
| `repetition` | Penalty lặp n-gram (repetition_n_grams=3, repetition_max_penalty=-1.0). |
| `cosine` | Cosine length reward (kết hợp accuracy + độ dài). |
| `soft_overlong` | Penalty khi generation vượt soft_max_length. |

Nhiều reward: `--reward_funcs grpo_accuracy format --reward_weights 1 0.2` (số weight = số reward).

**Config GRPO theo dataset (origin / low / med / high):**

- **Origin:** MODEL = checkpoint SFT origin, DATASET = `merged_grpo_data.jsonl` (hoặc file merge gsm8k+compmath), OUTPUT_DIR = `outputs/llama_origin_grpo`.
- **Low / Med / High:** MODEL = checkpoint SFT tương ứng (vd SFT trên `sft_low` → MODEL đó), DATASET = cùng file GRPO merged (thường vẫn `merged_grpo_data.jsonl` hoặc `train_grpo.jsonl`), OUTPUT_DIR = `outputs/llama_low_grpo`, `outputs/llama_med_grpo`, `outputs/llama_high_grpo`.

Ví dụ:

```bash
# GRPO từ SFT low
export MODEL=path/to/llama_3b_sft_low
export DATASET=/path/to/merged_grpo_data.jsonl
export OUTPUT_DIR=outputs/llama_low_grpo
bash scripts/train.sh
```

---

## Tóm tắt nhanh

| Giai đoạn | Framework | LoRA rank | LR | Max length | Dataset (train) |
|-----------|-----------|-----------|-----|-------------|------------------|
| SFT (low/med/high/origin) | LLaMA-Factory | 32 | 2e-4 | cutoff_len 3072 | sft_low / sft_medium / sft_high / origin |
| GRPO | MS-SWIFT | 8 (default), có thể 32 | 1e-4 (default) | 3072 | merged_grpo_data.jsonl (hoặc train_grpo.jsonl) |

Reward GRPO mặc định trong repo: `grpo_accuracy` (accuracy + repetition bonus). Có thể thêm `format`, `repetition`, … và `reward_weights` tương ứng.

## Dependencies

Root `requirements.txt` lists pandas/numpy/tqdm/jupyter/etc. **LLaMA-Factory** and **ms-swift** are installed separately (`pip install -e .` in each subfolder). Optional: `flash-attn` — install manually; it is not a normal single-line `requirements.txt` entry.
