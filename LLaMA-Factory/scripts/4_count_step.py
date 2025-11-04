#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import json
import os
from typing import Dict, List, Optional, Tuple

import fire
import torch
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments
from vllm import LLM, SamplingParams

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


def _is_prediction_complete(pred: Optional[str]) -> bool:
    """Kiểm tra xem prediction có hoàn chỉnh không (có chứa #### và câu trả lời sau đó)"""
    if not isinstance(pred, str):
        return False
    stripped = pred.strip()
    if not stripped:
        return False
    marker = "####"
    if marker not in stripped:
        return False
    suffix = stripped[stripped.rfind(marker) + len(marker):].strip()
    return bool(suffix)


def infer_vllm_direct(
    model_name_or_path: str,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 2048,
    skip_special_tokens: bool = True,
    default_system: Optional[str] = "",
    enable_thinking: bool = True,
    batch_size: int = 256,
    generations_per_sample: int = 5,
    
    # vLLM specific parameters
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    trust_remote_code: bool = True,
    dtype: str = "auto",
    seed: int = 42,
):
    """
    Inference trực tiếp với vLLM engine, sử dụng batch inference để tối ưu tốc độ.
    
    Args:
        model_name_or_path: Đường dẫn đến model
        dataset: Tên dataset
        dataset_dir: Thư mục chứa dataset
        template: Template cho prompt
        cutoff_len: Độ dài tối đa của input
        max_samples: Số lượng samples tối đa (None = toàn bộ)
        save_name: Tên file output
        temperature: Temperature cho sampling
        top_p: Top-p sampling
        max_new_tokens: Số tokens tối đa để generate
        skip_special_tokens: Bỏ qua special tokens khi decode
        default_system: System prompt mặc định
        enable_thinking: Bật chế độ thinking
        batch_size: Kích thước batch cho inference
        generations_per_sample: Số lượng generations cho mỗi sample
        gpu_memory_utilization: % GPU memory sử dụng
        max_model_len: Độ dài context tối đa của model
        trust_remote_code: Trust remote code khi load model
        dtype: Data type (auto, float16, bfloat16)
        seed: Random seed
    """
    
    print("=" * 70)
    print("[Config] vLLM Direct Inference")
    print(f"  model_name_or_path  = {model_name_or_path}")
    print(f"  dataset/dataset_dir = {dataset} / {dataset_dir}")
    print(f"  template            = {template}")
    print(f"  cutoff_len          = {cutoff_len}")
    print(f"  max_samples         = {max_samples}")
    print(f"  save_name           = {save_name}")
    print(f"  temperature/top_p   = {temperature} / {top_p}")
    print(f"  max_new_tokens      = {max_new_tokens}")
    print(f"  skip_special_tokens = {skip_special_tokens}")
    print(f"  default_system      = {repr(default_system)}")
    print(f"  enable_thinking     = {enable_thinking}")
    print(f"  batch_size          = {batch_size}")
    print(f"  generations/sample  = {generations_per_sample}")
    print(f"  gpu_memory_util     = {gpu_memory_utilization}")
    print(f"  max_model_len       = {max_model_len}")
    print(f"  dtype               = {dtype}")
    print(f"  seed                = {seed}")
    print("=" * 70)

    if generations_per_sample < 1:
        raise ValueError("generations_per_sample must be at least 1")
    
    # 1) Load tokenizer và dataset như LlamaFactory
    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            default_system=default_system,
            enable_thinking=enable_thinking,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False

    dataset_module = get_dataset(
        template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module
    )
    train_dataset = dataset_module["train_dataset"]

    # 2) Kiểm tra resume mode
    existing_generations: Dict[int, set] = {}
    empty_records: Dict[Tuple[int, int], dict] = {}
    resume_mode = os.path.exists(save_name)
    
    if resume_mode:
        print(f"[Resume] Found existing file {save_name}. Loading processed samples...")
        with open(save_name, "r", encoding="utf-8") as existing_file:
            for line in existing_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                sample_id = record.get("id")
                generation_index = record.get("generation_index")
                try:
                    sample_id = int(sample_id)
                    generation_index = int(generation_index)
                except (TypeError, ValueError):
                    continue
                
                predict_text = record.get("predict")
                if _is_prediction_complete(predict_text):
                    existing_generations.setdefault(sample_id, set()).add(generation_index)
                else:
                    empty_records[(sample_id, generation_index)] = record
        
        total_existing = sum(len(v) for v in existing_generations.values())
        print(f"[Resume] Loaded {len(existing_generations)} samples with {total_existing} generations.")
        if empty_records:
            print(f"[Resume] Found {len(empty_records)} empty predictions to refill.")

    # 3) Khởi tạo vLLM engine
    print("\n[vLLM] Initializing LLM engine...")
    llm = LLM(
        model=model_name_or_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        seed=seed,
    )
    print("[vLLM] Engine initialized successfully!")

    # 4) Chuẩn bị sampling params
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        seed=seed,
        skip_special_tokens=skip_special_tokens,
    )

    # 5) Batch inference
    open_mode = "a" if resume_mode else "w"
    refilled_records: Dict[Tuple[int, int], dict] = {}

    with open(save_name, open_mode, encoding="utf-8") as f:
        for i in tqdm(range(0, len(train_dataset), batch_size), desc="Batch Inference"):
            batch = train_dataset[i : min(i + batch_size, len(train_dataset))]

            # Thu thập các prompts cần inference
            batch_prompts: List[str] = []
            batch_labels: List[str] = []
            batch_sample_ids: List[int] = []
            batch_generation_indices: List[int] = []

            for j in range(len(batch["input_ids"])):
                sample_id = i + j
                existing_for_sample = existing_generations.get(sample_id, set())
                
                # Bỏ qua nếu đã đủ số generations
                if len(existing_for_sample) >= generations_per_sample:
                    continue

                # Tìm các generation index còn thiếu
                missing_indices = [
                    idx for idx in range(generations_per_sample) 
                    if idx not in existing_for_sample
                ]
                if not missing_indices:
                    continue

                # Decode prompt và label
                prompt_text = tokenizer.decode(
                    batch["input_ids"][j], 
                    skip_special_tokens=skip_special_tokens
                )
                label_ids = list(filter(lambda x: x != IGNORE_INDEX, batch["labels"][j]))
                label_text = tokenizer.decode(label_ids, skip_special_tokens=skip_special_tokens)

                # Thêm vào batch cho mỗi generation cần tạo
                for generation_idx in missing_indices:
                    batch_prompts.append(prompt_text)
                    batch_labels.append(label_text)
                    batch_sample_ids.append(sample_id)
                    batch_generation_indices.append(generation_idx)

            # Nếu không có gì để inference, skip
            if not batch_prompts:
                gc.collect()
                torch.cuda.empty_cache()
                continue

            # Chạy vLLM batch inference
            outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

            # Xử lý kết quả
            for sample_id, generation_idx, prompt_text, label_text, output in zip(
                batch_sample_ids,
                batch_generation_indices,
                batch_prompts,
                batch_labels,
                outputs,
            ):
                # Lấy text prediction từ output
                pred_text = output.outputs[0].text if output.outputs else ""
                
                # Tạo record
                record = {
                    "id": sample_id,
                    "generation_index": generation_idx,
                    "prompt": prompt_text,
                    "predict": pred_text,
                    "label": label_text,
                }
                
                key = (sample_id, generation_idx)
                is_complete = _is_prediction_complete(pred_text)

                # Xử lý empty records
                if key in empty_records:
                    if is_complete:
                        refilled_records[key] = record
                        existing_generations.setdefault(sample_id, set()).add(generation_idx)
                    else:
                        empty_records[key] = record
                        continue
                else:
                    # Ghi vào file
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if is_complete:
                        existing_generations.setdefault(sample_id, set()).add(generation_idx)
                    else:
                        empty_records[key] = record

            f.flush()
            gc.collect()
            torch.cuda.empty_cache()

    # 6) Refill empty records nếu có
    if refilled_records:
        print(f"\n[Resume] Refilling {len(refilled_records)} empty predictions...")
        tmp_output = f"{save_name}.tmp"
        with open(save_name, "r", encoding="utf-8") as src, \
             open(tmp_output, "w", encoding="utf-8") as dst:
            for line in src:
                original_line = line.rstrip("\n")
                if not original_line:
                    dst.write(line)
                    continue
                
                try:
                    record = json.loads(original_line)
                except json.JSONDecodeError:
                    dst.write(line)
                    continue
                
                sample_id = record.get("id")
                generation_index = record.get("generation_index")
                try:
                    key = (int(sample_id), int(generation_index))
                except (TypeError, ValueError):
                    dst.write(line)
                    continue
                
                replacement = refilled_records.get(key)
                if replacement:
                    dst.write(json.dumps(replacement, ensure_ascii=False) + "\n")
                else:
                    dst.write(line)
        
        os.replace(tmp_output, save_name)
        print(f"[Resume] Refilled {len(refilled_records)} predictions successfully!")

    print("\n" + "*" * 70)
    print("✓ Done! Results saved at:", save_name)
    print("*" * 70)


if __name__ == "__main__":
    fire.Fire(infer_vllm_direct)