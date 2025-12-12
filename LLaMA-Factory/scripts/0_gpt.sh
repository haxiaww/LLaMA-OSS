export CUDA_VISIBLE_DEVICES=2 #might have to rerun med cuz i forgot to change the mode

python3 0_gpt.py \
  --model_name_or_path=/mnt/dataset1/pretrained_fm/gpt-oss-20b \
  --dataset=/home/vlai-gpt-oss/LLAMA_OSS/LLaMA-OSS/LLaMA-Factory/data/competition_math_train_alpaca.jsonl \
  --save_name=compmath_train_high.jsonl \
  --batch_size=2048 \
  --generations_per_sample=5 \
  --temperature=1 \
  --max_new_tokens=2048 \
  --mode=high \
  --dataset_name=compmath \
  --instruction="Provide detailed reasoning with multiple approaches."

python3 0_gpt.py \
  --model_name_or_path=/mnt/dataset1/pretrained_fm/gpt-oss-20b \
  --dataset=/home/vlai-gpt-oss/LLAMA_OSS/LLaMA-OSS/LLaMA-Factory/data/gsm8k_train_alpaca.jsonl \
  --save_name=gms8k_train_high.jsonl \
  --batch_size=2048 \
  --generations_per_sample=5 \
  --temperature=1 \
  --max_new_tokens=2048 \
  --mode=high \
  --dataset_name=gsm8k \
  --instruction="Provide detailed reasoning with multiple approaches."