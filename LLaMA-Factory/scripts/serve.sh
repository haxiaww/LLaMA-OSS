reasoning_effort="${1:-high}"  # low, medium, high

case "$reasoning_effort" in
  low)
    max_num_seqs=256
    max_model_len=4096
    ;;
  medium)
    max_num_seqs=192
    max_model_len=6144
    ;;
  high)
    max_num_seqs=164
    max_model_len=10144
    ;;
  *)
    echo "reasoning_effort không hợp lệ: $reasoning_effort" >&2
    exit 1
    ;;
esac

CUDA_VISIBLE_DEVICES=0 vllm serve openai/gpt-oss-20b \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 20288 \
  --max-model-len $max_model_len \
  --max-num-seqs $max_num_seqs \