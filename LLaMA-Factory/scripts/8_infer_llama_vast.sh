CUDA_VISIBLE_DEVICES=1 python3 scripts/vllm_infer.py \
    --model_name_or_path output/llama_3b_sft_3mode \
    --dataset sft_grpo_val_high \
    --template llama3 \
    --batch_size 4096 \
    --default_system "" \
    --cutoff_len 2048 \
    --max_new_tokens 4096 \
    --seed 42 \
    --temperature 0.0 \
    --vllm_config '{"gpu_memory_utilization": 0.9, "max_model_len": 6144, "max_num_batched_tokens": 20480, "seed": 42}' \
    --save_name scripts/sft/8_infer/llama_3b_sft_3mode_0_high.jsonl \

CUDA_VISIBLE_DEVICES=1 python3 scripts/vllm_infer.py \
    --model_name_or_path output/llama_3b_sft_3mode \
    --dataset sft_grpo_val_high \
    --template llama3 \
    --batch_size 4096 \
    --default_system "" \
    --cutoff_len 2048 \
    --max_new_tokens 4096 \
    --seed 43 \
    --temperature 0.0 \
    --vllm_config '{"gpu_memory_utilization": 0.9, "max_model_len": 6144, "max_num_batched_tokens": 20480, "seed": 43}' \
    --save_name scripts/sft/8_infer/llama_3b_sft_3mode_1_high.jsonl \

CUDA_VISIBLE_DEVICES=1 python3 scripts/vllm_infer.py \
    --model_name_or_path output/llama_3b_sft_3mode \
    --dataset sft_grpo_val_high \
    --template llama3 \
    --batch_size 4096 \
    --default_system "" \
    --cutoff_len 2048 \
    --max_new_tokens 4096 \
    --seed 44 \
    --temperature 0.0 \
    --vllm_config '{"gpu_memory_utilization": 0.9, "max_model_len": 6144, "max_num_batched_tokens": 20480, "seed": 44}' \
    --save_name scripts/sft/8_infer/llama_3b_sft_3mode_2_high.jsonl \
