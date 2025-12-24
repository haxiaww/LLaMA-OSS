#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
#export USE_HF=1
#consider save_steps grad checkp reduces gpu warmup def is 0
#    --model_type llama3_2 \ Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3-2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct
#    --resume_from_checkpoint /workspace/LLaMA-OSS/outputs/llama_grpo/v5-20251124-110119/checkpoint-700 \
# KoiiVN/final_llama_3b_sft_low
# KoiiVN/final_llama_3b_sft_medium
# KoiiVN/final_llama_3b_sft_high
# KoiiVN/final_llama_3b_sft_raw
# train lại low

swift rlhf \
    --rlhf_type grpo \
    --model KoiiVN/final_llama_3b_sft_origin \
    --model_type llama3_2 \
    --dataset /home/vlai-gpt-oss/LLaMA-OSS/merged_grpo_data.jsonl \
    --per_device_train_batch_size 8 \
    --train_type lora \
    --gradient_accumulation_steps 2 \
    --max_steps 300 \
    --max_length 3072 \
    --loss_type dapo \
    --save_steps 100 \
    --logging_steps 10 \
    --reward_funcs grpo_accuracy \
    --reward_weights 1 \
    --output_dir ./outputs/llama_origin_grpo \
    --bf16 true \
    --gradient_checkpointing true \
    --warmup_ratio 0.05 \
    --save_total_limit 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --use_hf
echo "Training complete!"

# swift rlhf \
#     --rlhf_type grpo \
#     --model KoiiVN/final_llama_3b_sft_medium \
#     --model_type llama3_2 \
#     --dataset /home/vlai-gpt-oss/LLaMA-OSS/merged_grpo_data.jsonl \
#     --per_device_train_batch_size 8 \
#     --train_type lora \
#     --gradient_accumulation_steps 2 \
#     --max_steps 300 \
#     --max_length 3072 \
#     --loss_type dapo \
#     --save_steps 100 \
#     --logging_steps 10 \
#     --reward_funcs grpo_accuracy \
#     --reward_weights 1 \
#     --output_dir ./outputs/llama_med_grpo \
#     --bf16 true \
#     --gradient_checkpointing true \
#     --warmup_ratio 0.05 \
#     --save_total_limit 4 \
#     --num_generations 8 \
#     --temperature 1.0 \
#     --use_hf
# echo "Training complete!"

# swift rlhf \
#     --rlhf_type grpo \
#     --model KoiiVN/final_llama_3b_sft_high \
#     --model_type llama3_2 \
#     --dataset /home/vlai-gpt-oss/LLaMA-OSS/merged_grpo_data.jsonl \
#     --per_device_train_batch_size 8 \
#     --train_type lora \
#     --gradient_accumulation_steps 2 \
#     --max_steps 300 \
#     --max_length 3072 \
#     --loss_type dapo \
#     --save_steps 100 \
#     --logging_steps 10 \
#     --reward_funcs grpo_accuracy \
#     --reward_weights 1 \
#     --output_dir ./outputs/llama_high_grpo \
#     --bf16 true \
#     --gradient_checkpointing true \
#     --warmup_ratio 0.05 \
#     --save_total_limit 4 \
#     --num_generations 8 \
#     --temperature 1.0 \
#     --use_hf
# echo "Training complete!"

# swift rlhf \
#     --rlhf_type grpo \
#     --model KoiiVN/final_llama_3b_sft_low \
#     --model_type llama3_2 \
#     --dataset /home/vlai-gpt-oss/LLaMA-OSS/grpo_low.jsonl \
#     --per_device_train_batch_size 8 \
#     --train_type lora \
#     --gradient_accumulation_steps 2 \
#     --max_steps 300 \
#     --max_length 3072 \
#     --loss_type dapo \
#     --save_steps 100 \
#     --logging_steps 10 \
#     --reward_funcs grpo_accuracy \
#     --reward_weights 1 \
#     --output_dir ./outputs/llama_low_grpo \
#     --bf16 true \
#     --gradient_checkpointing true \
#     --warmup_ratio 0.05 \
#     --save_total_limit 4 \
#     --num_generations 8 \
#     --temperature 1.0 \
#     --use_hf
# echo "Training complete!"

# swift rlhf \
#     --rlhf_type grpo \
#     --model meta-llama/Llama-3.2-3B-Instruct \
#     --dataset /home/vlai-gpt-oss/LLaMA-OSS/compmath_grpo.jsonl,/home/vlai-gpt-oss/LLaMA-OSS/gsm8k_grpo.jsonl \
#     --per_device_train_batch_size 1 \
#     --train_type lora \
#     --max_steps 1000 \
#     --gradient_accumulation_steps 32 \
#     --max_length 3072 \
#     --loss_type dapo \
#     --save_steps 250 \
#     --logging_steps 10 \
#     --reward_funcs grpo_accuracy \
#     --reward_weights 1 \
#     --output_dir ./outputs/llama_base_grpo \
#     --bf16 true \
#     --gradient_checkpointing true \
#     --warmup_ratio 0.05 \
#     --save_total_limit 4 \
#     --use_vllm true \
#     --vllm_mode colocate \
#     --vllm_gpu_memory_utilization 0.6 \
#     --vllm_max_model_len 3072 \
#     --num_sample_generations 8 \
#     --temperature 1.0 \
#     --top_p 0.95 \
#     --use_hf
# echo "Training complete!"

#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
# export USE_HF=1
#consider save_steps grad checkp reduces gpu warmup def is 0
#    --model_type llama3_2 \ Qwen/qwen2_5_3b_instruct meta-llama/Llama-3-2-1B-Instruct
#    --resume_from_checkpoint /workspace/LLaMA-OSS/outputs/llama_grpo/v5-20251124-110119/checkpoint-700 \

# CUDA_VISIBLE_DEVICES=0
# USE_HF=1
# swift rlhf \
#     --rlhf_type grpo \
#     --model meta-llama/Llama-3.2-3B-Instruct \
#     --dataset train_grpo_formatted.jsonl \
#     --per_device_train_batch_size 1 \
#     --train_type lora \
#     --lora_rank 32 \
#     --lora_alpha 64 \
#     --max_steps 500 \
#     --gradient_accumulation_steps 32 \
#     --learning_rate 5e-7 \
#     --max_length 1024 \
#     --max_completion_length 2048 \
#     --vllm_max_model_len 3072 \
#     --loss_type dapo \
#     --epsilon_high 0.28 \
#     --epsilon 0.2 \
#     --save_steps 100 \
#     --logging_steps 5 \
#     --reward_funcs mode_adaptive \
#     --reward_weights 1 \
#     --output_dir ./outputs/llama_grpo \
#     --bf16 true \
#     --gradient_checkpointing true \
#     --warmup_ratio 0.05 \
#     --save_total_limit 2 \
#     --use_vllm true \
#     --vllm_gpu_memory_utilization 0.6 \
#     --num_sample_generations 8 \
#     --temperature 1.0 \
#     --top_p 0.95 \
#     --use_hf \
#     --report_to wandb \
#     --use_liger_kernel true \
#     --attn_impl flash_attn \
#     --sleep_level 1 \
#     --offload_model true \
#     --offload_optimizer true \


# echo "Training complete!"