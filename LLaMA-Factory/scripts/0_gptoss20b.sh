if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <reasoning_effort:{low,medium,high}> <dataset>" >&2
  exit 1
fi
reasoning_effort="$1"
dataset="$2"

case "$reasoning_effort" in
  low|medium|high) ;;
  *)
    echo "reasoning_effort không hợp lệ: $reasoning_effort (chọn low|medium|high)" >&2
    exit 1
    ;;
esac

case "$dataset" in
  gsm8k)
    case "$reasoning_effort" in
      low)
        concurrency=512
        max_new_tokens=1024
        save_name="gptoss20b_gsm8k_train_results_low.jsonl"
        openai_base_url=http://localhost:8002/v1
        ;;
      medium)
        concurrency=320
        max_new_tokens=2048
        save_name="gptoss20b_gsm8k_train_results_medium.jsonl"
        openai_base_url=http://localhost:8001/v1
        ;;
      high)
        concurrency=160
        max_new_tokens=4096
        save_name="gptoss20b_gsm8k_train_results_high.jsonl"
        openai_base_url=http://localhost:8000/v1
        ;;
      *) echo "reasoning_effort không hợp lệ: $reasoning_effort" >&2; exit 1 ;;
    esac
    dataset_name="gsm8k_train"
    ;;
  logiqa)
    case "$reasoning_effort" in
      low)
        concurrency=640
        max_new_tokens=1024
        save_name="gptoss20b_logiqa_train_results_low.jsonl"
        openai_base_url=http://localhost:8002/v1
        ;;
      medium)
        concurrency=320
        max_new_tokens=2048
        save_name="gptoss20b_logiqa_train_results_medium.jsonl"
        openai_base_url=http://localhost:8001/v1
        ;;
      high)
        concurrency=160
        max_new_tokens=4096
        save_name="gptoss20b_logiqa_train_results_high.jsonl"
        openai_base_url=http://localhost:8000/v1
        ;;
      *) echo "reasoning_effort không hợp lệ: $reasoning_effort" >&2; exit 1 ;;
    esac
    dataset_name="logiqa_train"
    ;;
  *)
    echo "dataset không hợp lệ: $dataset" >&2
    exit 1
    ;;
esac

python3 scripts/0_openai_responses_infer.py \
  --model_name_or_path /mnt/dataset1/pretrained_fm/gpt-oss-20b \
  --template gpt \
  --dataset_dir "data" \
  --cutoff_len 1024 \
  --temperature 1.0 \
  --top_p 1.0 \
  --skip_special_tokens False \
  --default_system "" \
  --enable_thinking True \
  --batch_size 512 \
  --generations_per_sample 5 \
  --dataset $dataset_name \
  --openai_base_url $openai_base_url \
  --save_name "$save_name" \
  --max_new_tokens "$max_new_tokens" \
  --reasoning_effort "$reasoning_effort" \
  --concurrency "$concurrency" \
