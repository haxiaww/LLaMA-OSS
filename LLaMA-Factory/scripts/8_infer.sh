device=$1
model_name_or_path="$2"
model_name=$3

# CUDA_VISIBLE_DEVICES=$1 python3 scripts/vllm_infer.py \
#     --model_name_or_path "$model_name_or_path" \
#     --dataset sft_grpo_val \
#     --template gpt \
#     --batch_size 1024 \
#     --default_system "" \
#     --cutoff_len 1024 \
#     --max_new_tokens 4096 \
#     --seed 42 \
#     --vllm_config '{"gpu_memory_utilization": 0.8, "max_model_len": 5120, "max_num_batched_tokens": 10240}' \
#     --save_name ${model_name_or_path}_tem1.jsonl \
#     --temperature 1.0 \


CUDA_VISIBLE_DEVICES=$1 python3 scripts/vllm_infer.py \
    --model_name_or_path "$model_name_or_path" \
    --dataset sft_grpo_val \
    --template qwen \
    --batch_size 2048 \
    --default_system "" \
    --cutoff_len 2048 \
    --max_new_tokens 8192 \
    --seed 42 \
    --vllm_config '{"gpu_memory_utilization": 0.9, "max_model_len": 10240, "max_num_batched_tokens": 20480}' \
    --save_name scripts/sft/8_infer/${model_name}_tem1_8192_10240.jsonl \
    --temperature 1.0 \
