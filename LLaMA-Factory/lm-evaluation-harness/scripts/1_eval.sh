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
  exit 1
fi

# Map known hub models; otherwise treat as local under output/
name=""
case "${pretrained_in}" in
  "meta-llama/Llama-3.2-3B-Instruct")
    # pretrained="meta-llama/${pretrained_in}"
    name="llama3_2_3b_base"
    ;;
  "qwen/Qwen-2.5-3B-Instruct")
    name="qwen2_5_3b_base"
    ;;
  *)
    name="${pretrained_in}"
    ;;
esac

# Output paths
mkdir -p results
timestamp="$(date +%Y%m%d-%H%M%S)"
out_json="results/${task}_${name}_4096_6144.json"

echo "Evaluating model: ${pretrained_in}"
echo "Task: ${task}"
echo "Device: ${device}"
echo "Output: ${out_json}"

CUDA_VISIBLE_DEVICES="${device}" lm_eval \
  --model vllm \
  --model_args "pretrained=${pretrained_in},tensor_parallel_size=1,dtype=bfloat16,max_model_len=6144,gpu_memory_utilization=0.9,trust_remote_code=True" \
  --tasks "${task}" \
  --batch_size auto \
  --gen_kwargs "max_gen_toks=4096" \
  --output_path "${out_json}" \
  --log_samples \
  --verbosity DEBUG \

