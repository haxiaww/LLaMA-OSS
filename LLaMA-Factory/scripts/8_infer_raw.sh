CUDA_VISIBLE_DEVICES=0 python3 scripts/vllm_infer.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct\
    --dataset sft_grpo_val_low \
    --template llama3 \
    --batch_size 4096 \
    --default_system "" \
    --cutoff_len 2048 \
    --max_new_tokens 4096 \
    --seed 42 \
    --vllm_config '{"gpu_memory_utilization": 0.95, "max_model_len": 6144, "max_num_batched_tokens": 20480}' \
    --save_name /workspace/LLaMA-OSS/val_raw/llama_grpo_low.jsonl \
    --temperature 1.0 \

CUDA_VISIBLE_DEVICES=0 python3 scripts/vllm_infer.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset sft_grpo_val_medium \
    --template llama3 \
    --batch_size 4096 \
    --default_system "" \
    --cutoff_len 2048 \
    --max_new_tokens 4096 \
    --seed 42 \
    --vllm_config '{"gpu_memory_utilization": 0.95, "max_model_len": 6144, "max_num_batched_tokens": 20480}' \
    --save_name /workspace/LLaMA-OSS/val_raw/llama_grpo_med.jsonl \
    --temperature 1.0 \

CUDA_VISIBLE_DEVICES=0 python3 scripts/vllm_infer.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset sft_grpo_val_high \
    --template llama3 \
    --batch_size 4096 \
    --default_system "" \
    --cutoff_len 2048 \
    --max_new_tokens 4096 \
    --seed 42 \
    --vllm_config '{"gpu_memory_utilization": 0.95, "max_model_len": 6144, "max_num_batched_tokens": 20480}' \
    --save_name /workspace/LLaMA-OSS/val_raw/llama_grpo_high.jsonl \
    --temperature 1.0 \
