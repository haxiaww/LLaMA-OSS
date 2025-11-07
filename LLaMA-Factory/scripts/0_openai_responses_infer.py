#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio, gc, json, math, os
from typing import Dict, Optional, List, Tuple
import re
import fire
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer
from openai import AsyncOpenAI, BadRequestError


def _is_prediction_complete(pred: Optional[str]) -> bool:
    if not isinstance(pred, str):
        return False
    
    stripped = pred.strip()
    if not stripped:
        return False
    
    # After JSON parsing: "\\boxed{10}" becomes '\boxed{10}' (single backslash)
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, stripped)
    # Consider complete if has at least one boxed expression with content
    return len(matches) > 0 and any(m.strip() for m in matches)


async def _one_call(
    client: AsyncOpenAI,
    model: str,
    prompt_text: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    reasoning_effort: Optional[str],
    pbar: Optional[tqdm] = None,
) -> Tuple[str, Optional[str], Optional[dict]]:
    async def call_responses() -> Tuple[str, Optional[str], Optional[dict]]:
        req_kwargs = {
            "model": model,
            "input": [{"role": "user", "content": prompt_text}],
            "temperature": temperature,
            "top_p": top_p or 1.0,
        }
        if max_output_tokens is not None:
            req_kwargs["max_output_tokens"] = max_output_tokens
        if reasoning_effort:
            req_kwargs["reasoning"] = {"effort": reasoning_effort}
        
        r = await client.responses.create(**req_kwargs)
        
        # Extract reasoning content and output text from Response object
        reasoning_text = ""
        output_text = ""
        
        if hasattr(r, 'output') and r.output:
            for item in r.output:
                # Extract reasoning content
                if hasattr(item, 'type') and item.type == 'reasoning':
                    if hasattr(item, 'content') and item.content:
                        for content_item in item.content:
                            if hasattr(content_item, 'type') and content_item.type == 'reasoning_text':
                                if hasattr(content_item, 'text'):
                                    reasoning_text = content_item.text
                
                # Extract final output text
                elif hasattr(item, 'type') and item.type == 'message':
                    if hasattr(item, 'content') and item.content:
                        for content_item in item.content:
                            if hasattr(content_item, 'type') and content_item.type == 'output_text':
                                if hasattr(content_item, 'text'):
                                    output_text = content_item.text
        
        # Extract token usage from Response.usage object
        usage_dict = None
        if hasattr(r, 'usage') and r.usage is not None:
            try:
                usage = r.usage
                input_tokens = getattr(usage, "input_tokens", None)
                output_tokens = getattr(usage, "output_tokens", None)
                
                # Extract reasoning_tokens from output_tokens_details
                reasoning_tokens = None
                if hasattr(usage, "output_tokens_details"):
                    output_details = usage.output_tokens_details
                    if output_details is not None:
                        reasoning_tokens = getattr(output_details, "reasoning_tokens", None)
                
                usage_dict = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "reasoning_tokens": reasoning_tokens,
                }
            except Exception as e:
                if pbar:
                    pbar.write(f"[Warning] Failed to extract usage: {e}")
                usage_dict = None
        
        # Update progress bar
        if pbar:
            pbar.update(1)
        
        return output_text.strip(), reasoning_text.strip(), usage_dict
    
    try:
        return await call_responses()
    except BadRequestError as err:
        if pbar:
            pbar.write(f"[Warning] BadRequestError for model {model}: {err}. Returning empty prediction.")
        if pbar:
            pbar.update(1)
        return "", "", None
    except Exception as e:
        if pbar:
            pbar.write(f"[Error] Unexpected error: {e}")
        if pbar:
            pbar.update(1)
        return "", "", None


async def _infer_batch_async(
    client: AsyncOpenAI,
    model: str,
    prompts: List[str],
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    reasoning_effort: Optional[str],
    concurrency: int,
    generations_per_prompt: int = 1,
    pbar: Optional[tqdm] = None,
) -> List[List[Tuple[str, str, Optional[dict]]]]:
    if not prompts:
        return []
    
    sem = asyncio.Semaphore(concurrency)
    
    async def worker(p: str) -> Tuple[str, str, Optional[dict]]:
        async with sem:
            return await _one_call(
                client,
                model,
                p,
                temperature,
                top_p,
                max_output_tokens,
                reasoning_effort,
                pbar,
            )
    
    tasks = []
    prompt_indices = []
    for prompt_idx, prompt_text in enumerate(prompts):
        for _ in range(generations_per_prompt):
            tasks.append(asyncio.create_task(worker(prompt_text)))
            prompt_indices.append(prompt_idx)
    
    raw_results = await asyncio.gather(*tasks)
    
    preds: List[List[Tuple[str, str, Optional[dict]]]] = [[] for _ in prompts]
    for prompt_idx, pred in zip(prompt_indices, raw_results):
        preds[prompt_idx].append(pred)
    
    return preds


def infer_via_openai_responses_fast(
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
    batch_size: int = 1024,
    generations_per_sample: int = 5,
    # OpenAI-compatible API (vLLM)
    openai_base_url: str = "http://localhost:8000/v1",
    openai_api_key: Optional[str] = "not-needed",
    reasoning_effort: Optional[str] = None,
    # Tối ưu tốc độ
    concurrency: int = 64,
):
    
    print("=" * 70)
    print("[Config] Client params:")
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
    print(f"  openai_base_url     = {openai_base_url}")
    print(f"  reasoning_effort    = {reasoning_effort}")
    print(f"  concurrency         = {concurrency}")
    print("=" * 70)
    
    if generations_per_sample < 1:
        raise ValueError("generations_per_sample must be at least 1")
    
    # 1) Giữ pipeline decode prompt/label như LlamaFactory
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
    
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)
    train_dataset = dataset_module["train_dataset"]
    
    print(f"\n[Dataset] Total samples in dataset: {len(train_dataset)}")
    print(f"[Dataset] Generations per sample: {generations_per_sample}")
    print(f"[Dataset] Total generations needed: {len(train_dataset) * generations_per_sample}")
    
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
        
        total_existing_generations = sum(len(v) for v in existing_generations.values())
        print(f"[Resume] Loaded {len(existing_generations)} samples covering {total_existing_generations} generations.")
        if empty_records:
            print(f"[Resume] Detected {len(empty_records)} empty predictions that will be refilled.")
    
    open_mode = "a" if resume_mode else "w"
    refilled_records: Dict[Tuple[int, int], dict] = {}
    
    # Calculate total pending generations
    total_pending = 0
    for i in range(len(train_dataset)):
        existing_for_sample = existing_generations.get(i, set())
        missing_count = generations_per_sample - len(existing_for_sample)
        if missing_count > 0:
            total_pending += missing_count
    
    print(f"\n[Progress] Pending generations to process: {total_pending}")
    print(f"[Progress] Already completed: {len(train_dataset) * generations_per_sample - total_pending}")
    
    async def _process_batches(f_handle):
        client = AsyncOpenAI(
            base_url=openai_base_url, 
            api_key=openai_api_key or "not-needed",
            timeout=60
        )
        
        # Create main progress bars
        batch_pbar = tqdm(
            total=math.ceil(len(train_dataset) / batch_size),
            desc="Batches",
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        generation_pbar = tqdm(
            total=total_pending,
            desc="Generations",
            position=1,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        # Stats tracking
        total_input_tokens = 0
        total_output_tokens = 0
        total_reasoning_tokens = 0
        completed_count = 0
        failed_count = 0
        
        for i in range(0, len(train_dataset), batch_size):
            batch = train_dataset[i : min(i + batch_size, len(train_dataset))]
            
            pending_prompts = []
            pending_labels = []
            pending_sample_ids = []
            pending_generation_indices = []
            
            for j in range(len(batch["input_ids"])):
                sample_id = i + j
                existing_for_sample = existing_generations.get(sample_id, set())
                
                if len(existing_for_sample) >= generations_per_sample:
                    continue
                
                missing_indices = [idx for idx in range(generations_per_sample) if idx not in existing_for_sample]
                if not missing_indices:
                    continue
                
                prompt_text = tokenizer.decode(batch["input_ids"][j], skip_special_tokens=skip_special_tokens)
                label_ids = list(filter(lambda x: x != IGNORE_INDEX, batch["labels"][j]))
                label_text = tokenizer.decode(label_ids, skip_special_tokens=skip_special_tokens)
                
                for generation_idx in missing_indices:
                    pending_prompts.append(prompt_text)
                    pending_labels.append(label_text)
                    pending_sample_ids.append(sample_id)
                    pending_generation_indices.append(generation_idx)
            
            if not pending_prompts:
                batch_pbar.update(1)
                gc.collect()
                continue
            
            # Update batch description with current batch info
            batch_pbar.set_postfix({
                'samples': f'{i}/{len(train_dataset)}',
                'pending': len(pending_prompts)
            })
            
            preds_by_prompt = await _infer_batch_async(
                client=client,
                model=model_args.model_name_or_path,
                prompts=pending_prompts,
                temperature=generating_args.temperature,
                top_p=generating_args.top_p or 1.0,
                max_output_tokens=generating_args.max_new_tokens,
                reasoning_effort=reasoning_effort,
                concurrency=concurrency,
                generations_per_prompt=1,
                pbar=generation_pbar,
            )
            
            for sample_id, generation_idx, prompt_text, label_text, pred_list in zip(
                pending_sample_ids,
                pending_generation_indices,
                pending_prompts,
                pending_labels,
                preds_by_prompt,
            ):
                # pred_list contains a single tuple: (output_text, reasoning_text, usage_dict)
                if pred_list and len(pred_list) > 0:
                    pred_text, reasoning_text, usage_dict = pred_list[0]
                else:
                    pred_text, reasoning_text, usage_dict = "", "", None
                
                record = {
                    "id": sample_id,
                    "generation_index": generation_idx,
                    "prompt": prompt_text,
                    "predict": pred_text,
                    "label": label_text,
                }
                
                # Thêm trường reasoning nếu có
                if reasoning_text:
                    record["reasoning"] = reasoning_text
                
                # Attach token usage fields when available
                if isinstance(usage_dict, dict):
                    if usage_dict.get("input_tokens") is not None:
                        record["input_tokens"] = usage_dict.get("input_tokens")
                        total_input_tokens += usage_dict.get("input_tokens")
                    if usage_dict.get("output_tokens") is not None:
                        record["output_tokens"] = usage_dict.get("output_tokens")
                        total_output_tokens += usage_dict.get("output_tokens")
                    if usage_dict.get("reasoning_tokens") is not None:
                        record["reasoning_tokens"] = usage_dict.get("reasoning_tokens")
                        total_reasoning_tokens += usage_dict.get("reasoning_tokens")
                
                key = (sample_id, generation_idx)
                is_complete = _is_prediction_complete(pred_text)
                
                if is_complete:
                    completed_count += 1
                else:
                    failed_count += 1
                
                # Update stats in generation progress bar
                generation_pbar.set_postfix({
                    'completed': completed_count,
                    'failed': failed_count,
                    'tokens': f'{total_output_tokens:,}'
                })
                
                if key in empty_records:
                    if is_complete:
                        refilled_records[key] = record
                        existing_generations.setdefault(sample_id, set()).add(generation_idx)
                    else:
                        empty_records[key] = record
                        continue
                else:
                    f_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if is_complete:
                        existing_generations.setdefault(sample_id, set()).add(generation_idx)
                    else:
                        empty_records[key] = record
            
            f_handle.flush()
            batch_pbar.update(1)
            gc.collect()
        
        batch_pbar.close()
        generation_pbar.close()
        
        # Print final statistics
        print("\n" + "=" * 70)
        print("[Statistics] Final Results:")
        print(f"  Total generations: {completed_count + failed_count}")
        print(f"  Completed: {completed_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Success rate: {completed_count / (completed_count + failed_count) * 100:.2f}%")
        print(f"  Total input tokens: {total_input_tokens:,}")
        print(f"  Total output tokens: {total_output_tokens:,}")
        print(f"  Total reasoning tokens: {total_reasoning_tokens:,}")
        if total_output_tokens > 0:
            print(f"  Avg output tokens/generation: {total_output_tokens / (completed_count + failed_count):.2f}")
        if total_reasoning_tokens > 0:
            print(f"  Avg reasoning tokens/generation: {total_reasoning_tokens / (completed_count + failed_count):.2f}")
        print("=" * 70)
        
        close_coro = getattr(client, "close", None)
        if close_coro:
            result = close_coro()
            if asyncio.iscoroutine(result):
                await result
    
    with open(save_name, open_mode, encoding="utf-8") as f:
        asyncio.run(_process_batches(f))
    
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
    fire.Fire(infer_via_openai_responses_fast)