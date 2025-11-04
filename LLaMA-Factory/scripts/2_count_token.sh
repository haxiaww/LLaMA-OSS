#!/bin/bash
# Script đếm tokens cho file JSONL predictions
reasoning_effort="${1:-medium}"
# Input file

case "$reasoning_effort" in
  low)
    INPUT_FILE="scripts/new_dataset/1_fbl/gptoss20b_gsm8k_train_results_low_fbl.jsonl"
    ;;
  medium)
    INPUT_FILE="scripts/new_dataset/1_fbl/gptoss20b_gsm8k_train_results_medium_fbl.jsonl"
    ;;
  high)
    INPUT_FILE="scripts/new_dataset/1_fbl/gptoss20b_gsm8k_train_results_high_fbl.jsonl"
    ;;
  *)
    echo "reasoning_effort không hợp lệ: $reasoning_effort" >&2
    exit 1
    ;;
esac


# Check if file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File không tồn tại: $INPUT_FILE"
    echo "Usage: $0 <input_file.jsonl>"
    exit 1
fi

# Auto-generate output filename
BASE_NAME="${INPUT_FILE%.jsonl}"
OUTPUT_FILE="${BASE_NAME}_tokens.jsonl"

echo "========================================"
echo "Counting tokens for: $INPUT_FILE"
echo "Output will be saved to: $OUTPUT_FILE"
echo "========================================"

python3 scripts/2_count_token.py count \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --model_name_or_path /mnt/dataset1/pretrained_fm/gpt-oss-20b \
    --template gpt \
    --skip_special_tokens False \
    --overwrite_existing False

echo ""
echo "========================================"
echo "✓ Done! Check output at: $OUTPUT_FILE"
echo "========================================"