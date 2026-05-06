# Evaluation Guide

Benchmarking uses **EleutherAI lm-evaluation-harness** with the **vLLM** model backend. The repo provides `scripts/eval.sh` (works from any cwd; results go under `<repo>/results/all_tasks/`).

## Prerequisites

- `lm_eval` on your PATH (install from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness); this repo does not pin it in root `requirements.txt`).
- GPU with enough memory for `--model_args` below (bf16, `max_model_len=4096`).
- Your **merged / LoRA / full** checkpoint path passed as the second argument.

## Environment overrides (`eval.sh`)

| Env | Default | Role |
|-----|---------|------|
| `LM_EVAL_TASKS` | `minerva_math500,gsm8k` | lm-eval task list |
| `TP_SIZE` | `1` | vLLM `tensor_parallel_size` |
| `MAX_MODEL_LEN` | `4096` | vLLM context |
| `GPU_MEM_UTIL` | `0.9` | vLLM GPU memory fraction |

## `scripts/eval.sh` arguments

```bash
bash scripts/eval.sh [CUDA_DEVICES] [MODEL_PATH] [NAME_TAG]
```

| Arg | Default | Meaning |
|-----|---------|---------|
| 1 | `0` | `CUDA_VISIBLE_DEVICES` |
| 2 | *(required)* | `pretrained=` in vLLM args — HF id or local path |
| 3 | `eval` | Suffix in the output filename under `results/all_tasks/` |

Example:

```bash
bash scripts/eval.sh 0 /path/to/merged_or_adapter_checkpoint my_run
```

Outputs go under `results/all_tasks/` (see script for exact filename pattern).

## What runs

- **Tasks**: defaults `minerva_math500,gsm8k` (override with `LM_EVAL_TASKS`)
- **Few-shot**: `--num_fewshot 0` (0-shot in the template)
- **Generation**: `max_gen_toks=2048`, `temperature=0`, `do_sample=False`
- **Chat**: `--apply_chat_template`
- **Samples**: `--log_samples` (large logs; disable if you only need aggregate scores)

## Adapting

- Change `--tasks` for other lm-eval task names.
- If you use **LoRA adapters**, lm-eval vLLM integration may need a **merged** full weights directory — check current harness docs for `pretrained` + adapter options.
- Multi-GPU: set **`TP_SIZE`** (env) to match visible devices, e.g. `TP_SIZE=2 bash scripts/eval.sh 0,1 …`. Override tasks with **`LM_EVAL_TASKS`**, context with **`MAX_MODEL_LEN`** / **`GPU_MEM_UTIL`** if needed.
