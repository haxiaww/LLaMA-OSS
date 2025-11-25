#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export USE_HF=1
#consider save_steps grad checkp reduces gpu warmup def is 0
#    --model_type llama3_2 \ Qwen/qwen2_5_3b_instruct meta-llama/Llama-3-2-1B-Instruct
#    --resume_from_checkpoint /workspace/LLaMA-OSS/outputs/llama_grpo/v5-20251124-110119/checkpoint-700 \

swift rlhf \
    --rlhf_type grpo \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dataset train_grpo_formatted.jsonl \
    --per_device_train_batch_size 1 \
    --train_type lora \
    --max_steps 1000 \
    --gradient_accumulation_steps 32 \
    --learning_rate 5e-7 \
    --max_length 3072 \
    --loss_type grpo \
    --save_steps 100 \
    --logging_steps 10 \
    --reward_funcs mode_adaptive \
    --reward_weights 1 \
    --output_dir ./outputs/llama_grpo \
    --bf16 true \
    --gradient_checkpointing true \
    --warmup_ratio 0.05 \
    --save_total_limit 3 \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_max_model_len 3072 \
    --num_sample_generations 4 \
    --temperature 1.0 \
    --top_p 0.95 \
    --use_hf
echo "Training complete!"