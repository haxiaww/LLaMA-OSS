#!/bin/bash

# Script chạy vLLM direct inference
# Usage: ./run_vllm_direct.sh [config]
# config: low, medium, high (default: medium)

config="${1:-medium}"

case "$config" in
  low)
    save_name="scripts/new_dataset/3_fbs/gsm8k_train_low.jsonl"
    ;;
  medium)
    save_name="scripts/new_dataset/3_fbs/gsm8k_train_medium.jsonl"
    ;;
  high)
    save_name="scripts/new_dataset/3_fbs/gsm8k_train_high.jsonl"
    ;;
  *)
    echo "Config không hợp lệ: $config" >&2
    echo "Sử dụng: $0 [low|medium|high]" >&2
    exit 1
    ;;
esac

VLLM_USE_V1=1
VLLM_USE_FLASHINFER_SAMPLER=1 
VLLM_LOGGING_LEVEL=DEBUG 
CUDA_VISIBLE_DEVICES=1 python3 scripts/4_count_step.py \
  --model_name_or_path /mnt/dataset1/pretrained_fm/gpt-oss-20b \
  --template gpt \
  --dataset high_reason_gsm8k_train \
  --dataset_dir "data" \
  --cutoff_len 1024 \
  --temperature 1.0 \
  --top_p 1.0 \
  --skip_special_tokens False \
  --default_system "" \
  --enable_thinking True \
  --batch_size 4096 \
  --generations_per_sample 3 \
  --save_name "$save_name" \
  --max_new_tokens 1536 \
  --gpu_memory_utilization 0.9 \
  --max_model_len 2048 \
  --trust_remote_code True \
  --dtype auto \
  --seed 42 \

echo "=========================================="
echo "✓ Done!"
echo "Results saved to: $save_name"
echo "=========================================="