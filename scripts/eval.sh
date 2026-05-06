#!/usr/bin/env bash
# lm-eval + vLLM (default: minerva_math500, gsm8k). Safe from any cwd.
# Optional env: LM_EVAL_TASKS, TP_SIZE, MAX_MODEL_LEN, GPU_MEM_UTIL
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

device="${1:-0}"
model_path="${2:-}"
name="${3:-eval}"

: "${LM_EVAL_TASKS:=minerva_math500,gsm8k}"
: "${TP_SIZE:=1}"
: "${MAX_MODEL_LEN:=4096}"
: "${GPU_MEM_UTIL:=0.9}"

if [[ -z "$model_path" ]]; then
  echo "Usage: $0 [CUDA_DEVICES] MODEL_PATH [NAME_TAG]" >&2
  echo "  Optional env: LM_EVAL_TASKS TP_SIZE MAX_MODEL_LEN GPU_MEM_UTIL" >&2
  exit 1
fi

out_dir="${REPO_ROOT}/results/all_tasks"
mkdir -p "$out_dir"
out_json="${out_dir}/${name}_0shot_2048_4096"

echo "Model: ${model_path}"
echo "Tasks: ${LM_EVAL_TASKS} | GPU: ${device} | TP: ${TP_SIZE}"
echo "Output: ${out_json}.json"

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export CUDA_VISIBLE_DEVICES="${device}"

model_args="pretrained=${model_path},tensor_parallel_size=${TP_SIZE},dtype=bfloat16,max_model_len=${MAX_MODEL_LEN},gpu_memory_utilization=${GPU_MEM_UTIL},trust_remote_code=True"

lm_eval \
  --model vllm \
  --model_args "${model_args}" \
  --tasks "${LM_EVAL_TASKS}" \
  --batch_size auto \
  --output_path "${out_json}.json" \
  --log_samples \
  --verbosity DEBUG \
  --apply_chat_template \
  --gen_kwargs "max_gen_toks=2048,temperature=0,do_sample=False" \
  --seed 1234 \
  --num_fewshot 0
