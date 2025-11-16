#!/usr/bin/env bash
# eval.sh
set -euo pipefail

device="${1:-0}"         # e.g., "0" or "0,1"
pretrained_in="${2:-}"   # e.g., Llama-3.2-3B-Instruct | Qwen-2.5-3B-Instruct | llama3_2_3b_sft_concat_sorted
task="${3:-}"            # e.g., logiqa

if [[ -z "${pretrained_in}" || -z "${task}" ]]; then
  echo "Usage: $0 <device> <pretrained_or_local_name> <task> [tensor_parallel_size]"
  echo "Examples:"
  echo "  $0 0 Llama-3.2-3B-Instruct gsm8k"
  echo "  $0 0,1 llama3_2_3b_sft_concat_sorted logiqa 2"
  exit 1
fi

# Map known hub models; otherwise treat as local under output/
name="" ; pretrained=""
case "${pretrained_in}" in
  "meta-llama/Llama-3.2-3B-Instruct")
    # pretrained="meta-llama/${pretrained_in}"
    pretrained="${pretrained_in}"
    name="llama3_2_3b_base"
    ;;
  "qwen/Qwen-2.5-3B-Instruct")
    pretrained="${pretrained_in}"
    name="qwen2_5_3b_base"
    ;;
  *)
    pretrained="${pretrained_in}"
    name="${pretrained_in}"
    ;;
esac

CUDA_VISIBLE_DEVICES=1 lm_eval \
  --model vllm \
  --model_args "pretrained=${pretrained},tensor_parallel_size=1,dtype=bfloat16,max_model_len=10240,gpu_memory_utilization=0.8,trust_remote_code=True" \
  --tasks logiqa \
  --batch_size auto \
  --gen_kwargs "max_gen_toks=8192" \
  --output_path "${out_json}" \
  --log_samples \
  --verbosity DEBUG
