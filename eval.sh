#!/usr/bin/env bash
# ./eval.sh 2 /home/vlai-gpt-oss/LLaMA-OSS/LLaMA-Factory/putput/grpo_base
# eval.sh
set -euo pipefail

device="${1:-0}"         # e.g., "0" or "0,1"
pretrained_in="${2:-}"   # e.g., Llama-3.2-3B-Instruct | Qwen-2.5-3B-Instruct | llama3_2_3b_sft_concat_sorted

# Map known hub models; otherwise treat as local under output/
name=""
case "${pretrained_in}" in
  "meta-llama/Llama-3.2-3B-Instruct")
    echo "Do not use this script for a base model."
    exit 1
    ;;
  "Qwen/Qwen2.5-3B-Instruct")
    echo "Do not use this script for a base model."
    exit 1
    ;;
  "google/gemma-2-2b-it")
    echo "Do not use this script for a base model."
    exit 1
    ;;
  *)
    name="${pretrained_in}"
    ;;
esac

# Output paths
mkdir -p results
timestamp="$(date +%Y%m%d-%H%M%S)"
out_json="results/all_tasks/${name}"

echo "Evaluating model: ${pretrained_in}"
echo "Task: minerva_math500, gsm8k"
echo "Device: ${device}"
echo "Output: ${out_json}"

VLLM_USE_FLASHINFER=0 VLLM_DISABLE_CUDA_GRAPH=1 CUDA_VISIBLE_DEVICES="${device}" lm_eval \
  --model vllm \
  --model_args "pretrained=${pretrained_in},tensor_parallel_size=1,dtype=float16,max_model_len=6144,gpu_memory_utilization=0.9,trust_remote_code=True" \
  --tasks minerva_math500,gsm8k \
  --batch_size auto \
  --output_path "${out_json}.json" \
  --log_samples \
  --verbosity DEBUG \
  --apply_chat_template \
  --num_fewshot 0 \
  --gen_kwargs "max_gen_toks=4096" \
