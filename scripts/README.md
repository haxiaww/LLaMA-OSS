# Scripts (repo root)

| File | Purpose |
|------|---------|
| `eval.sh` | lm-eval + vLLM — see [EVALUATION.md](../common/md/EVALUATION.md) |
| `train.sh` | MS-SWIFT GRPO — see [CONFIG.md](../common/md/CONFIG.md) |
| `convert_data.py` | JSONL prep subcommands — see [USAGE.md](../common/md/USAGE.md) |
| `convert_data.sh` | `python3 scripts/convert_data.py "$@"` with `REPO_ROOT` set |
| `convert_data.ipynb` | Interactive wrapper around `convert_data.py` |

Run from **any cwd**; scripts resolve the repo root from their own path.
