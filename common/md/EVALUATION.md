# Evaluation Guide

Benchmarking uses **EleutherAI lm-evaluation-harness** with the **vLLM** model backend. The repo provides `eval.sh` at the **repository root**.

## Prerequisites

- `lm_eval` on your PATH (install from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness); this repo does not pin it in root `requirements.txt`).
- GPU with enough memory for `--model_args` below (bf16, `max_model_len=4096`).
- Your **merged / LoRA / full** checkpoint path passed as the second argument.

## `eval.sh` arguments

```bash
./eval.sh [CUDA_DEVICES] [MODEL_PATH] [NAME_TAG]
```

| Arg | Default | Meaning |
|-----|---------|---------|
| 1 | `0` | `CUDA_VISIBLE_DEVICES` |
| 2 | *(required for real runs)* | `pretrained=` in vLLM args — HF id or local path |
| 3 | *(optional)* | Suffix for output folder naming |

Example:

```bash
bash eval.sh 0 /path/to/merged_or_adapter_checkpoint my_run
```

Outputs go under `results/all_tasks/` (see script for exact filename pattern).

## What runs

- **Tasks**: `minerva_math500,gsm8k`
- **Few-shot**: `--num_fewshot 0` (0-shot in the template)
- **Generation**: `max_gen_toks=2048`, `temperature=0`, `do_sample=False`
- **Chat**: `--apply_chat_template`
- **Samples**: `--log_samples` (large logs; disable if you only need aggregate scores)

## Adapting

- Change `--tasks` for other lm-eval task names.
- If you use **LoRA adapters**, lm-eval vLLM integration may need a **merged** full weights directory — check current harness docs for `pretrained` + adapter options.
- For multi-GPU tensor parallel, adjust `tensor_parallel_size` inside `eval.sh` to match visible devices.
