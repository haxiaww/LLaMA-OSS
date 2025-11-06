VLLM_USE_V1=1
VLLM_USE_FLASHINFER_SAMPLER=1 
VLLM_LOGGING_LEVEL=DEBUG 
CUDA_VISIBLE_DEVICES=0 vllm serve openai/gpt-oss-20b \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 32768 \
  --max-model-len 9120 \
  --max-num-seqs 320 \
