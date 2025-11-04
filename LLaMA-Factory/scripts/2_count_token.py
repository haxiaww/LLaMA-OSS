#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để đếm số tokens trong file JSONL predictions và thêm trường "output_token"
"""
import json
import os
from typing import Optional
import fire
from tqdm import tqdm
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


def count_tokens_in_jsonl(
    input_file: str,
    output_file: Optional[str] = None,
    model_name_or_path: str = "/mnt/dataset1/pretrained_fm/gpt-oss-20b",
    template: str = "gpt",
    skip_special_tokens: bool = False,
    batch_size: int = 2048,
    overwrite_existing: bool = False,
):
    """
    Đọc file JSONL, tính số tokens cho trường 'gpt_reasoning', và thêm trường 'output_token'.
    
    Args:
        input_file: Đường dẫn đến file JSONL input
        output_file: Đường dẫn đến file JSONL output (mặc định: input_file với suffix _with_tokens)
        model_name_or_path: Đường dẫn đến model để load tokenizer
        template: Template name cho tokenizer
        skip_special_tokens: Có bỏ qua special tokens khi đếm không
        batch_size: Số records xử lý cùng lúc (để tiết kiệm memory)
        overwrite_existing: Có ghi đè output_token nếu đã tồn tại không
    """
    
    print("=" * 70)
    print("[Config] Token Counting Script")
    print(f"  input_file          = {input_file}")
    print(f"  model_name_or_path  = {model_name_or_path}")
    print(f"  template            = {template}")
    print(f"  skip_special_tokens = {skip_special_tokens}")
    print(f"  batch_size          = {batch_size}")
    print(f"  overwrite_existing  = {overwrite_existing}")
    print("=" * 70)
    
    # Kiểm tra file input tồn tại
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file không tồn tại: {input_file}")
    
    # Xác định output file
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_with_tokens.jsonl"
    
    print(f"  output_file         = {output_file}")
    print("=" * 70)
    
    # Load tokenizer
    print("\n[1/3] Loading tokenizer...")
    model_args, _, _, _ = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            template=template,
        )
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    print(f"✓ Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    # Đọc và xử lý file
    print(f"\n[2/3] Reading input file: {input_file}")
    
    # Đếm tổng số dòng để hiển thị progress
    total_lines = 0
    with open(input_file, "r", encoding="utf-8") as f:
        for _ in f:
            total_lines += 1
    
    print(f"✓ Found {total_lines} records")
    
    # Xử lý và ghi file
    print(f"\n[3/3] Processing and writing to: {output_file}")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        
        for line in tqdm(infile, total=total_lines, desc="Counting tokens"):
            line = line.strip()
            if not line:
                outfile.write("\n")
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"\n⚠ Warning: Không thể parse JSON tại dòng {processed_count + 1}: {e}")
                outfile.write(line + "\n")
                error_count += 1
                continue
            
            # Kiểm tra nếu đã có output_token và không overwrite
            if "output_token" in record and not overwrite_existing:
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                skipped_count += 1
                continue
            
            # Lấy text cần đếm tokens từ 'gpt_reasoning' (mặc định mới)
            predict_text = record.get("gpt_reasoning", "")
            
            if not predict_text:
                record["output_token"] = 0
            else:
                # Tokenize và đếm
                token_ids = tokenizer.encode(
                    predict_text,
                    add_special_tokens=not skip_special_tokens
                )
                record["output_token"] = len(token_ids)
            
            # Ghi record mới
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed_count += 1
    
    print("\n" + "=" * 70)
    print("[Summary]")
    print(f"  Total records:      {total_lines}")
    print(f"  Processed:          {processed_count}")
    print(f"  Skipped (existing): {skipped_count}")
    print(f"  Errors:             {error_count}")
    print(f"\n✓ Output saved to: {output_file}")
    print("=" * 70)


def count_tokens_for_field(
    input_file: str,
    field_name: str = "gpt_reasoning",
    output_field: str = "output_token",
    output_file: Optional[str] = None,
    model_name_or_path: str = "/mnt/dataset1/pretrained_fm/gpt-oss-20b",
    template: str = "gpt",
    skip_special_tokens: bool = False,
    overwrite_existing: bool = False,
):
    """
    Version linh hoạt hơn: đếm tokens cho bất kỳ field nào.
    
    Args:
        input_file: Đường dẫn đến file JSONL input
        field_name: Tên field cần đếm tokens (mặc định: "gpt_reasoning")
        output_field: Tên field để lưu số tokens (mặc định: "output_token")
        output_file: Đường dẫn đến file JSONL output
        model_name_or_path: Đường dẫn đến model để load tokenizer
        template: Template name cho tokenizer
        skip_special_tokens: Có bỏ qua special tokens khi đếm không
        overwrite_existing: Có ghi đè nếu output_field đã tồn tại không
    """
    
    print("=" * 70)
    print("[Config] Flexible Token Counting")
    print(f"  input_file          = {input_file}")
    print(f"  field_name          = {field_name}")
    print(f"  output_field        = {output_field}")
    print(f"  model_name_or_path  = {model_name_or_path}")
    print(f"  template            = {template}")
    print(f"  skip_special_tokens = {skip_special_tokens}")
    print(f"  overwrite_existing  = {overwrite_existing}")
    print("=" * 70)
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file không tồn tại: {input_file}")
    
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_with_{output_field}.jsonl"
    
    print(f"  output_file         = {output_file}")
    print("=" * 70)
    
    # Load tokenizer
    print("\n[1/3] Loading tokenizer...")
    model_args, _, _, _ = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            template=template,
        )
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    print(f"✓ Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    # Đếm lines
    print(f"\n[2/3] Reading input file: {input_file}")
    total_lines = sum(1 for _ in open(input_file, "r", encoding="utf-8"))
    print(f"✓ Found {total_lines} records")
    
    # Process
    print(f"\n[3/3] Processing and writing to: {output_file}")
    
    processed_count = 0
    skipped_count = 0
    missing_field_count = 0
    error_count = 0
    
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        
        for line in tqdm(infile, total=total_lines, desc=f"Counting tokens for '{field_name}'"):
            line = line.strip()
            if not line:
                outfile.write("\n")
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"\n⚠ Warning: Không thể parse JSON: {e}")
                outfile.write(line + "\n")
                error_count += 1
                continue
            
            # Skip nếu đã có và không overwrite
            if output_field in record and not overwrite_existing:
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                skipped_count += 1
                continue
            
            # Lấy text từ field
            text = record.get(field_name, "")
            
            if field_name not in record:
                missing_field_count += 1
                record[output_field] = 0
            elif not text:
                record[output_field] = 0
            else:
                token_ids = tokenizer.encode(
                    text,
                    add_special_tokens=not skip_special_tokens
                )
                record[output_field] = len(token_ids)
            
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed_count += 1
    
    print("\n" + "=" * 70)
    print("[Summary]")
    print(f"  Total records:       {total_lines}")
    print(f"  Processed:           {processed_count}")
    print(f"  Skipped (existing):  {skipped_count}")
    print(f"  Missing field:       {missing_field_count}")
    print(f"  Errors:              {error_count}")
    print(f"\n✓ Output saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    fire.Fire({
        "count": count_tokens_in_jsonl,
        "count_field": count_tokens_for_field,
    })
