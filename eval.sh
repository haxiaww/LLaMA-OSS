#!/usr/bin/env bash
# eval.sh
set -euo pipefail

device="${1:-0}"         # e.g., "0" or "0,1"
model_path="${2:-}"  
name="${3:-}"

# Output paths
mkdir -p results
timestamp="$(date +%Y%m%d-%H%M%S)"
out_json="results/all_tasks/${name}_0shot_2048_4096"

echo "Evaluating model: ${model_path}"
echo "Task: minerva_math500,gsm8k"
echo "Device: ${device}"
echo "Output: ${out_json}"

VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES="${device}" lm_eval \
  --model vllm \
  --model_args "pretrained=${model_path},tensor_parallel_size=1,dtype=bfloat16,max_model_len=4096,gpu_memory_utilization=0.9,trust_remote_code=True" \
  --tasks minerva_math500,gsm8k \
  --batch_size auto \
  --output_path "${out_json}.json" \
  --log_samples \
  --verbosity DEBUG \
  --apply_chat_template \
  --gen_kwargs "max_gen_toks=2048,temperature=0,do_sample=False" \
  --seed 1234 \
  --num_fewshot 0 \