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
        max_num_seqs=512
        max_model_len=2048
        max_num_batched_tokens=16384
        port=8002
        ;;
      medium)
        max_num_seqs=320
        max_model_len=3072
        max_num_batched_tokens=16384
        port=8001
        ;;
      high)
        max_num_seqs=160
        max_model_len=5120
        max_num_batched_tokens=16384
        port=8000
        ;;
      *) echo "reasoning_effort không hợp lệ: $reasoning_effort" >&2; exit 1 ;;
    esac
    ;;
  logiqa)
    case "$reasoning_effort" in
      low)
        max_num_seqs=640
        max_model_len=2048
        max_num_batched_tokens=16384
        port=8002
        ;;
      medium)
        max_num_seqs=320
        max_model_len=3072
        max_num_batched_tokens=16384
        port=8001
        ;;
      high)
        max_num_seqs=160
        max_model_len=5120
        max_num_batched_tokens=16384
        port=8000
        ;;
      *) echo "reasoning_effort không hợp lệ: $reasoning_effort" >&2; exit 1 ;;
    esac
    ;;
  *)
    echo "dataset không hợp lệ: $dataset" >&2
    exit 1
    ;;
esac

VLLM_USE_V1=1
VLLM_USE_FLASHINFER_SAMPLER=1 
VLLM_LOGGING_LEVEL=DEBUG 
CUDA_VISIBLE_DEVICES=1 vllm serve /mnt/dataset1/pretrained_fm/gpt-oss-20b \
  --host 0.0.0.0 --port $port \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens $max_num_batched_tokens \
  --max-model-len $max_model_len \
  --max-num-seqs $max_num_seqs \