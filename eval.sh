lm_eval \
  --model vllm \
  --model_args "pretrained=/workspace/LLaMA-OSS/model/llama_grpo_sft_3mode_4,tensor_parallel_size=1,dtype=bfloat16,max_model_len=6144,gpu_memory_utilization=0.9,trust_remote_code=True" \
  --tasks "gsm8k,logiqa,minerva_math500" \
  --batch_size auto \
  --gen_kwargs "max_gen_toks=4096" \
  --output_path "./val_output" \
  --log_samples \
  --verbosity DEBUG \