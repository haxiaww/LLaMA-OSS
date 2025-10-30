#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio, gc, json, math, os
from typing import Optional, List, Tuple

import fire
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer

from openai import AsyncOpenAI


async def _one_call(
    client: AsyncOpenAI,
    model: str,
    prompt_text: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    reasoning_effort: str,
    stream: bool,
) -> str:
    if stream:
        # Stream để giảm TTFT; kết quả cuối vẫn ghép lại thành text
        out = []
        stream_obj = await client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt_text}],
            reasoning={"effort": reasoning_effort},
            temperature=temperature,
            top_p=top_p or 1.0,
            max_output_tokens=max_output_tokens,
            stream=True,
        )
        async for chunk in stream_obj:
            # Responses stream: chunk có thể mang text trong output_text_delta
            txt = getattr(chunk, "output_text_delta", None)
            if txt:
                out.append(txt)
        return "".join(out).strip()
    else:
        r = await client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt_text}],
            reasoning={"effort": reasoning_effort},
            temperature=temperature,
            top_p=top_p or 1.0,
            max_output_tokens=max_output_tokens,
        )
        return getattr(r, "output_text", "").strip()


async def _infer_batch_async(
    client: AsyncOpenAI,
    model: str,
    prompts: List[str],
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    reasoning_effort: str,
    concurrency: int,
    stream: bool,
    generations_per_prompt: int = 1,
) -> List[List[str]]:
    
    if not prompts:
        return []
    sem = asyncio.Semaphore(concurrency)

    async def worker(p: str) -> str:
        async with sem:
            return await _one_call(
                client, model, p, temperature, top_p, max_output_tokens, reasoning_effort, stream
            )

    tasks = []
    prompt_indices = []
    for prompt_idx, prompt_text in enumerate(prompts):
        for _ in range(generations_per_prompt):
            tasks.append(asyncio.create_task(worker(prompt_text)))
            prompt_indices.append(prompt_idx)
    raw_results = await asyncio.gather(*tasks)

    preds: List[List[str]] = [[] for _ in prompts]
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
    reasoning_effort: str = "low",  # low | medium | high

    # Tối ưu tốc độ
    concurrency: int = 64,   # số request song song bên client (mỗi batch)
    stream: bool = False,    # True nếu muốn giảm TTFT; throughput tương đương
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
    print(f"  stream              = {stream}")
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

    client = AsyncOpenAI(base_url=openai_base_url, api_key=openai_api_key or "not-needed")

    existing_generations = {}
    resume_mode = os.path.exists(save_name)
    if resume_mode:
        print(f"[Resume] Found existing file {save_name}. Loading processed samples to resume...")
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
                existing_generations.setdefault(sample_id, set()).add(generation_index)
        total_existing_generations = sum(len(v) for v in existing_generations.values())
        print(f"[Resume] Loaded {len(existing_generations)} samples covering {total_existing_generations} generations.")

    open_mode = "a" if resume_mode else "w"

    # Ghi incrementally để không giữ toàn bộ vào RAM
    with open(save_name, open_mode, encoding="utf-8") as f:
        for i in tqdm(range(0, len(train_dataset), batch_size), desc="Processing batched inference (async)"):
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
                gc.collect()
                continue

            # chạy song song trong batch
            preds_by_prompt = asyncio.run(_infer_batch_async(
                client=client,
                model=model_args.model_name_or_path,
                prompts=pending_prompts,
                temperature=generating_args.temperature,
                top_p=generating_args.top_p or 1.0,
                max_output_tokens=generating_args.max_new_tokens,
                reasoning_effort=reasoning_effort,
                concurrency=concurrency,
                stream=stream,
                generations_per_prompt=1,
            ))

            # ghi ra file ngay
            for sample_id, generation_idx, prompt_text, label_text, pred_list in zip(
                pending_sample_ids,
                pending_generation_indices,
                pending_prompts,
                pending_labels,
                preds_by_prompt,
            ):
                pred = pred_list[0] if pred_list else ""
                record = {
                    "id": sample_id,
                    "generation_index": generation_idx,
                    "prompt": prompt_text,
                    "predict": pred,
                    "label": label_text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                existing_generations.setdefault(sample_id, set()).add(generation_idx)

            f.flush()
            gc.collect()

    print("*" * 70)
    print("Done. Results saved at", save_name)
    print("*" * 70)


if __name__ == "__main__":
    fire.Fire(infer_via_openai_responses_fast)
