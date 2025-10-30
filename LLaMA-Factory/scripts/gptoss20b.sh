reasoning_effort="${1:-low}"  # low, medium, high

case "$reasoning_effort" in
  low)
    concurrency=256
    max_new_tokens=2048
    ;;
  medium)
    concurrency=128
    max_new_tokens=4096
    ;;
  high)
    concurrency=64
    max_new_tokens=8096
    ;;
  *)
    echo "reasoning_effort không hợp lệ: $reasoning_effort" >&2
    exit 1
    ;;
esac

python3 scripts/openai_responses_infer.py \
  --model_name_or_path openai/gpt-oss-20b \
  --template gpt \
  --dataset gsm8k_train \
  --dataset_dir "data" \
  --save_name "gptoss20b_gsm8k_train_results_low.jsonl" \
  --cutoff_len 2048 \
  --temperature 1.0 \
  --top_p 1.0 \
  --skip_special_tokens False \
  --default_system "" \
  --enable_thinking True \
  --batch_size 1024 \
  --openai_base_url http://localhost:8000/v1 \
  --stream False \
  --max_new_tokens "$max_new_tokens" \
  --reasoning_effort "$reasoning_effort" \
  --concurrency "$concurrency" \
  --generations_per_sample 10 \