#!/usr/bin/env bash
# eval.sh
set -euo pipefail

device="${1:-0}"         # e.g., "0" or "0,1"
pretrained_in="${2:-}"   # e.g., Llama-3.2-3B-Instruct | Qwen-2.5-3B-Instruct | llama3_2_3b_sft_concat_sorted

# Map known hub models; otherwise treat as local under output/
name=""
case "${pretrained_in}" in
  "meta-llama/Llama-3.2-3B-Instruct")
    name="llama3_2_3b_base"
    ;;
  "Qwen/Qwen2.5-3B-Instruct")
    name="qwen2_5_3b_base"
    ;;
  "google/gemma-2-2b-it")
    name="gemma2_2b_base"
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
echo "Task: logiqa, minerva_math500, gsm8k"
echo "Device: ${device}"
echo "Output: ${out_json}"

CUDA_VISIBLE_DEVICES="${device}" lm_eval \
  --model vllm \
  --model_args "pretrained=${pretrained_in},tensor_parallel_size=2,dtype=bfloat16,max_model_len=6144,gpu_memory_utilization=0.5,trust_remote_code=True" \
  --tasks logiqa,minerva_math500,gsm8k \
  --batch_size auto \
  --output_path "${out_json}_low.json" \
  --log_samples \
  --verbosity DEBUG \
  --system_instruction "Respond concisely with minimal reasoning.\n" \
  --apply_chat_template \
  --wandb_args "project=lm-eval-harness-integration,name='${name}_low'" \
  --limit 1 \
  --gen_kwargs "max_gen_toks=4096" \


CUDA_VISIBLE_DEVICES="${device}" lm_eval \
  --model vllm \
  --model_args "pretrained=${pretrained_in},tensor_parallel_size=1,dtype=bfloat16,max_model_len=6144,gpu_memory_utilization=0.5,trust_remote_code=True" \
  --tasks logiqa,minerva_math500,gsm8k \
  --batch_size auto \
  --output_path "${out_json}_medium.json" \
  --log_samples \
  --verbosity DEBUG \
  --system_instruction "Solve step-by-step.\n" \
  --apply_chat_template \
  --wandb_args "project=lm-eval-harness-integration,name=${name}_medium" \
  --limit 1 \
  --gen_kwargs "max_gen_toks=4096" \


CUDA_VISIBLE_DEVICES="${device}" lm_eval \
  --model vllm \
  --model_args "pretrained=${pretrained_in},tensor_parallel_size=1,dtype=bfloat16,max_model_len=6144,gpu_memory_utilization=0.5,trust_remote_code=True" \
  --tasks logiqa,minerva_math500,gsm8k \
  --batch_size auto \
  --output_path "${out_json}_high.json" \
  --log_samples \
  --verbosity DEBUG \
  --system_instruction "Think deeply, verify, and self-correct.\n" \
  --apply_chat_template \
  --wandb_args "project=lm-eval-harness-integration,name='${name}_high'" \
  --limit 1 \
  --gen_kwargs "max_gen_toks=4096" \
