device=$1
model_name_or_path="$2"

CUDA_VISIBLE_DEVICES=1 python3 scripts/vllm_infer.py \
    --model_name_or_path "$model_name_or_path" \
    --dataset sft_grpo_val \
    --template llama3 \
    --batch_size 2048 \
    --default_system "" \
    --cutoff_len 1024 \
    --max_new_tokens 4096 \
    --seed 42 \
    --vllm_config '{"gpu_memory_utilization": 0.9, "max_model_len": 5120, "max_num_batched_tokens": 16384}' \
    --save_name ${model_name_or_path}_tem0.jsonl \
    --temperature 0.0 \

CUDA_VISIBLE_DEVICES=$device python3 scripts/vllm_infer.py \
    --model_name_or_path "$model_name_or_path" \
    --dataset sft_grpo_val \
    --template llama3 \
    --batch_size 2048 \
    --default_system "" \
    --cutoff_len 1024 \
    --max_new_tokens 4096 \
    --seed 42 \
    --vllm_config '{"gpu_memory_utilization": 0.9, "max_model_len": 5120, "max_num_batched_tokens": 16384}' \
    --save_name ${model_name_or_path}_tem1.jsonl \
    --temperature 1.0
