#!/usr/bin/env bash
# MS-SWIFT GRPO (LoRA). Override via env; defaults are one sensible preset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

: "${CUDA_VISIBLE_DEVICES:=2}"
: "${MODEL:=KoiiVN/final_llama_3b_sft_origin}"
: "${MODEL_TYPE:=llama3_2}"
: "${DATASET:=${REPO_ROOT}/merged_grpo_data.jsonl}"
: "${OUTPUT_DIR:=${REPO_ROOT}/outputs/llama_origin_grpo}"

export CUDA_VISIBLE_DEVICES

use_hf=(--use_hf)
[[ "${USE_HF:-1}" == "0" ]] && use_hf=()

swift rlhf \
  --rlhf_type grpo \
  --model "$MODEL" \
  --model_type "$MODEL_TYPE" \
  --dataset "$DATASET" \
  --per_device_train_batch_size 8 \
  --train_type lora \
  --gradient_accumulation_steps 2 \
  --max_steps 300 \
  --max_length 3072 \
  --loss_type dapo \
  --save_steps 100 \
  --logging_steps 10 \
  --reward_funcs grpo_accuracy \
  --reward_weights 1 \
  --output_dir "$OUTPUT_DIR" \
  --bf16 true \
  --gradient_checkpointing true \
  --warmup_ratio 0.05 \
  --save_total_limit 4 \
  --num_generations 8 \
  --temperature 1.0 \
  "${use_hf[@]}"
