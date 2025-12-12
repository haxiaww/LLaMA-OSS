#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure vLLM Inference with <think> Tags
No API server needed - uses vLLM LLM class directly
"""

import gc
import json
import os
import re
from typing import Dict, List, Optional, Tuple
import fire
from tqdm import tqdm
from vllm import LLM, SamplingParams


def _is_prediction_complete(pred: Optional[str]) -> bool:
    """Check if prediction contains a complete boxed answer."""
    if not isinstance(pred, str):
        return False
    
    stripped = pred.strip()
    if not stripped:
        return False
    
    # Check for \boxed{...} pattern
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, stripped)
    return len(matches) > 0 and any(m.strip() for m in matches)


def load_dataset_from_jsonl(
    dataset_path: str,
    max_samples: Optional[int] = None
) -> List[Dict]:
    """Load dataset from JSONL file.
    
    Expected format per line:
    {
        "prompt": "user prompt text",
        "response": "ground truth answer (optional)"
    }
    """
    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Normalize field names - support both 'response' and 'label'
                if 'response' in data and 'label' not in data:
                    data['label'] = data['response']
                samples.append(data)
            except json.JSONDecodeError:
                continue
    
    if max_samples:
        samples = samples[:max_samples]
    
    return samples


def format_prompt_with_think(
    prompt: str, 
    system_prompt: Optional[str] = None,
    enable_thinking: bool = True
) -> str:
    """Format prompt - just use the prompt as-is since it already has instructions."""
    # The prompt already contains instructions, so just return it
    if system_prompt:
        return f"{system_prompt}\n\n{prompt}"
    else:
        return prompt


def extract_thinking_and_answer(text: str) -> Tuple[str, str]:
    """Extract thinking content and final answer from response.
    
    Args:
        text: Full response text with <think> tags
    
    Returns:
        Tuple of (thinking_content, answer_content)
    """
    # Extract content inside <think> tags
    think_pattern = r'<think>(.*?)</think>'
    think_matches = re.findall(think_pattern, text, re.DOTALL)
    thinking = "\n".join(think_matches).strip() if think_matches else ""
    
    # Remove <think> tags to get remaining answer
    answer = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()
    
    return thinking, answer


def infer_vllm_pure(
    model_name_or_path: str,
    dataset: str,
    dataset_dir: str = "data",
    save_name: str = "generated_predictions.jsonl",
    template: str = "default",
    cutoff_len: int = 1024,
    max_samples: Optional[int] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    max_new_tokens: int = 2048,
    skip_special_tokens: bool = False,
    default_system: Optional[str] = "",
    enable_thinking: bool = True,
    batch_size: int = 2048,
    generations_per_sample: int = 5,
    # Output format parameters
    mode: str = "low",
    dataset_name: str = "compmath",
    instruction: str = "Respond concisely with minimal reasoning.",
    # vLLM specific
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    trust_remote_code: bool = True,
    dtype: str = "auto",
    seed: int = 42,
):
    """Run inference using pure vLLM (no API server needed).
    
    Args:
        model_name_or_path: Path to model
        dataset: Dataset path or name. Can be:
                 - Full path: /path/to/data.jsonl
                 - Dataset name: competition_math_train_alpaca (looks in dataset_dir)
        dataset_dir: Directory containing datasets (default: "data")
        save_name: Output file path
        template: Template type (not used in pure vLLM mode)
        cutoff_len: Maximum input length
        max_samples: Limit number of samples
        temperature: Sampling temperature
        top_p: Top-p sampling
        top_k: Top-k sampling
        max_new_tokens: Maximum output tokens
        skip_special_tokens: Whether to skip special tokens
        default_system: System prompt
        enable_thinking: Enable <think> tags
        batch_size: Batch size for inference
        generations_per_sample: Number of generations per sample
        mode: Reasoning mode label (e.g., "low", "medium", "high")
        dataset_name: Dataset identifier for output (e.g., "compmath")
        instruction: Instruction text for the task
        tensor_parallel_size: Number of GPUs
        gpu_memory_utilization: GPU memory fraction
        max_model_len: Maximum model length
        trust_remote_code: Trust remote code
        dtype: Data type
        seed: Random seed
    """
    
    print("=" * 70)
    print("[Config] Pure vLLM Inference Parameters:")
    print(f"  model_name_or_path  = {model_name_or_path}")
    print(f"  dataset             = {dataset}")
    print(f"  dataset_dir         = {dataset_dir}")
    print(f"  save_name           = {save_name}")
    print(f"  template            = {template} (ignored in pure vLLM)")
    print(f"  cutoff_len          = {cutoff_len}")
    print(f"  max_samples         = {max_samples}")
    print(f"  temperature/top_p   = {temperature} / {top_p}")
    print(f"  top_k               = {top_k}")
    print(f"  max_new_tokens      = {max_new_tokens}")
    print(f"  skip_special_tokens = {skip_special_tokens}")
    print(f"  default_system      = {repr(default_system)}")
    print(f"  enable_thinking     = {enable_thinking}")
    print(f"  batch_size          = {batch_size}")
    print(f"  generations/sample  = {generations_per_sample}")
    print(f"  mode                = {mode}")
    print(f"  dataset_name        = {dataset_name}")
    print(f"  instruction         = {instruction}")
    print(f"  tensor_parallel     = {tensor_parallel_size}")
    print(f"  gpu_memory_util     = {gpu_memory_utilization}")
    print(f"  max_model_len       = {max_model_len}")
    print(f"  seed                = {seed}")
    print("=" * 70)
    
    if generations_per_sample < 1:
        raise ValueError("generations_per_sample must be at least 1")
    
    # Build dataset path
    # Support both full paths and dataset names
    if os.path.isabs(dataset) or os.path.exists(dataset):
        # Dataset is a full path
        dataset_path = dataset
    else:
        # Dataset is a name, construct path with dataset_dir
        dataset_path = os.path.join(dataset_dir, f"{dataset}.jsonl")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    print(f"\n[Dataset] Loading from {dataset_path}...")
    dataset_samples = load_dataset_from_jsonl(dataset_path, max_samples)
    print(f"[Dataset] Total samples: {len(dataset_samples)}")
    print(f"[Dataset] Generations per sample: {generations_per_sample}")
    print(f"[Dataset] Total generations needed: {len(dataset_samples) * generations_per_sample}")
    
    # Initialize vLLM
    print(f"\n[vLLM] Initializing LLM...")
    
    # Prepare LLM kwargs
    llm_kwargs = {
        "model": model_name_or_path,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
        "dtype": dtype,
        "seed": seed,
    }
    
    # Add max_model_len only if specified
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    
    # Disable problematic quantization features to avoid triton errors
    try:
        llm_kwargs["disable_custom_all_reduce"] = True
    except:
        pass
    
    try:
        llm = LLM(**llm_kwargs)
        print(f"[vLLM] Model loaded successfully")
    except Exception as e:
        print(f"[Error] Failed to initialize vLLM: {e}")
        print("\n[Info] Trying with additional compatibility settings...")
        # Try with enforce_eager mode to avoid compilation issues
        llm_kwargs["enforce_eager"] = True
        try:
            llm = LLM(**llm_kwargs)
            print(f"[vLLM] Model loaded successfully (eager mode)")
        except Exception as e2:
            print(f"[Error] Still failed: {e2}")
            raise
    
    # Create sampling params
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        n=generations_per_sample,  # Number of generations per prompt
        seed=seed,
    )
    
    print(f"\n[Sampling] Parameters:")
    print(f"  temperature: {temperature}")
    print(f"  top_p: {top_p}")
    print(f"  top_k: {top_k}")
    print(f"  max_tokens: {max_new_tokens}")
    print(f"  n (generations): {generations_per_sample}")
    
    # Resume logic
    existing_generations: Dict[int, set] = {}
    empty_records: Dict[Tuple[int, int], dict] = {}
    
    resume_mode = os.path.exists(save_name)
    if resume_mode:
        print(f"\n[Resume] Found existing file {save_name}. Loading processed samples to resume...")
        with open(save_name, "r", encoding="utf-8") as existing_file:
            for line in existing_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Support both old and new format
                if "_meta" in record:
                    sample_id = record["_meta"].get("id")
                    generation_index = record["_meta"].get("generation_index")
                else:
                    # Try to extract from combined_index
                    combined_index = record.get("combined_index")
                    if combined_index is not None:
                        sample_id = combined_index // generations_per_sample
                        generation_index = combined_index % generations_per_sample
                    else:
                        # Old format fallback
                        sample_id = record.get("id")
                        generation_index = record.get("generation_index")
                
                try:
                    sample_id = int(sample_id)
                    generation_index = int(generation_index)
                except (TypeError, ValueError):
                    continue
                
                # Check if response is complete
                response_text = record.get("response", record.get("predict", ""))
                if _is_prediction_complete(response_text):
                    existing_generations.setdefault(sample_id, set()).add(generation_index)
                else:
                    empty_records[(sample_id, generation_index)] = record
        
        total_existing_generations = sum(len(v) for v in existing_generations.values())
        print(f"[Resume] Loaded {len(existing_generations)} samples covering {total_existing_generations} generations.")
        if empty_records:
            print(f"[Resume] Detected {len(empty_records)} empty predictions that will be refilled.")
    
    open_mode = "a" if resume_mode else "w"
    refilled_records: Dict[Tuple[int, int], dict] = {}
    
    # Calculate total pending generations
    total_pending = 0
    samples_to_process = []
    for i, sample in enumerate(dataset_samples):
        existing_for_sample = existing_generations.get(i, set())
        missing_count = generations_per_sample - len(existing_for_sample)
        if missing_count > 0:
            total_pending += missing_count
            samples_to_process.append((i, sample))
    
    print(f"\n[Progress] Pending generations to process: {total_pending}")
    print(f"[Progress] Already completed: {len(dataset_samples) * generations_per_sample - total_pending}")
    
    if total_pending == 0:
        print("\n[Info] All generations already complete!")
        return
    
    # Process in batches
    with open(save_name, open_mode, encoding="utf-8") as f:
        # Stats tracking
        total_completed = 0
        total_failed = 0
        
        for batch_start in tqdm(range(0, len(samples_to_process), batch_size), desc="Batches"):
            batch = samples_to_process[batch_start:batch_start + batch_size]
            
            # Prepare prompts
            prompts = []
            prompt_metadata = []  # Track (sample_id, sample_data)
            
            for sample_id, sample in batch:
                existing_for_sample = existing_generations.get(sample_id, set())
                if len(existing_for_sample) >= generations_per_sample:
                    continue
                
                prompt_text = sample.get("prompt", "")
                formatted_prompt = format_prompt_with_think(
                    prompt_text, 
                    default_system if default_system else None,
                    enable_thinking
                )
                
                prompts.append(formatted_prompt)
                prompt_metadata.append((sample_id, sample))
            
            if not prompts:
                continue
            
            # Generate with vLLM
            print(f"\n[Batch] Processing {len(prompts)} prompts...")
            outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
            
            # Process outputs
            for (sample_id, sample), output in tqdm(
                zip(prompt_metadata, outputs),
                total=len(outputs),
                desc="Processing outputs"
            ):
                prompt_text = output.prompt
                label_text = sample.get("label", "")
                
                existing_for_sample = existing_generations.get(sample_id, set())
                missing_indices = [idx for idx in range(generations_per_sample) if idx not in existing_for_sample]
                
                # Process each generation
                for generation_idx, completion_output in enumerate(output.outputs[:len(missing_indices)]):
                    if generation_idx >= len(missing_indices):
                        break
                    
                    actual_gen_idx = missing_indices[generation_idx]
                    pred_text = completion_output.text
                    
                    # Extract thinking and answer from the prediction
                    thinking_text, answer_text = extract_thinking_and_answer(pred_text)
                    
                    # Format response with <think> tags
                    if thinking_text:
                        response = f"<think>{thinking_text}</think> {answer_text}"
                    else:
                        response = answer_text
                    
                    # Estimate reasoning tokens (rough approximation: 4 chars per token)
                    reasoning_tokens = len(thinking_text) / 4.0 if thinking_text else 0.0
                    
                    # Store the record with your desired format
                    record = {
                        "prompt": prompt_text,
                        "response": response,
                        "mode": mode,
                        "reasoning_tokens": reasoning_tokens,
                        "label": label_text,
                        "dataset": dataset_name,
                        "instruction": instruction,
                        "combined_index": sample_id * generations_per_sample + actual_gen_idx,
                    }
                    
                    # Add optional fields
                    if hasattr(completion_output, 'cumulative_logprob'):
                        record["cumulative_logprob"] = completion_output.cumulative_logprob
                    
                    # Add metadata for tracking
                    record["_meta"] = {
                        "id": sample_id,
                        "generation_index": actual_gen_idx,
                    }
                    
                    key = (sample_id, actual_gen_idx)
                    is_complete = _is_prediction_complete(response)  # Check formatted response
                    
                    if is_complete:
                        total_completed += 1
                    else:
                        total_failed += 1
                    
                    # Handle empty records refilling
                    if key in empty_records:
                        if is_complete:
                            refilled_records[key] = record
                            existing_generations.setdefault(sample_id, set()).add(actual_gen_idx)
                        else:
                            empty_records[key] = record
                            continue
                    else:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        if is_complete:
                            existing_generations.setdefault(sample_id, set()).add(actual_gen_idx)
                        else:
                            empty_records[key] = record
            
            f.flush()
            gc.collect()
        
        # Print final statistics
        print("\n" + "=" * 70)
        print("[Statistics] Final Results:")
        print(f"  Total generations: {total_completed + total_failed}")
        print(f"  Completed: {total_completed}")
        print(f"  Failed: {total_failed}")
        if (total_completed + total_failed) > 0:
            print(f"  Success rate: {total_completed / (total_completed + total_failed) * 100:.2f}%")
        print("=" * 70)
    
    # Refill empty records if any
    if refilled_records:
        print(f"\n[Resume] Refilling {len(refilled_records)} empty predictions...")
        tmp_output = f"{save_name}.tmp"
        with open(save_name, "r", encoding="utf-8") as src, open(tmp_output, "w", encoding="utf-8") as dst:
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
                
                # Extract sample_id and generation_index from new or old format
                if "_meta" in record:
                    sample_id = record["_meta"].get("id")
                    generation_index = record["_meta"].get("generation_index")
                else:
                    combined_index = record.get("combined_index")
                    if combined_index is not None:
                        sample_id = combined_index // generations_per_sample
                        generation_index = combined_index % generations_per_sample
                    else:
                        sample_id = record.get("id")
                        generation_index = record.get("generation_index")
                
                key = None
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
        print(f"[Resume] Successfully refilled {len(refilled_records)} empty predictions.")
    
    print("\n" + "*" * 70)
    print(f"Done! Results saved at: {save_name}")
    print("*" * 70 + "\n")


if __name__ == "__main__":
    fire.Fire(infer_vllm_pure)