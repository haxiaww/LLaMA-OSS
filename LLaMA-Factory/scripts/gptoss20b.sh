reasoning_effort="${1:-high}"  # low, medium, high

case "$reasoning_effort" in
  low)
    concurrency=256
    max_new_tokens=2048
    save_name="gptoss20b_gsm8k_train_results_low.jsonl"
    ;;
  medium)
    concurrency=192
    max_new_tokens=4096
    save_name="gptoss20b_gsm8k_train_results_medium.jsonl"
    ;;
  high)
    concurrency=192
    max_new_tokens=8096
    save_name="gptoss20b_gsm8k_train_results_high.jsonl"
    ;;
  *)
    echo "reasoning_effort không hợp lệ: $reasoning_effort" >&2
    exit 1
    ;;
esac

python3 scripts/openai_responses_infer.py \
  --model_name_or_path /mnt/dataset1/pretrained_fm/gpt-oss-20b \
  --template gpt \
  --dataset gsm8k_train \
  --dataset_dir "data" \
  --cutoff_len 2048 \
  --temperature 1.0 \
  --top_p 1.0 \
  --skip_special_tokens False \
  --default_system "" \
  --enable_thinking True \
  --batch_size 1024 \
  --openai_base_url http://localhost:8000/v1 \
  --stream False \
  --save_name "$save_name" \
  --max_new_tokens "$max_new_tokens" \
  --reasoning_effort "$reasoning_effort" \
  --concurrency "$concurrency" \
  --generations_per_sample 10 \