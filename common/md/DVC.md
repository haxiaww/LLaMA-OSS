# DVC (data + artifacts)

Large paths are tracked with [DVC](https://dvc.org), not Git blobs:

| Path | DVC file |
|------|-----------|
| `LLaMA-Factory/data/` | `LLaMA-Factory/data.dvc` |
| `outputs/` | `outputs.dvc` |
| `merged_grpo_data.jsonl` | `merged_grpo_data.dvc` |
| `train_grpo.jsonl` | `train_grpo.dvc` |

Git stores only the `.dvc` files + hashes; file bytes live under `.dvc/cache` locally and on your **DVC remote** after `dvc push`.

## Install

```bash
pip install "dvc>=3.0"
# optional cloud extras, e.g. AWS S3:
# pip install "dvc[s3]"
```

(`requirements.txt` includes `dvc` for this repo.)

## Fresh clone

```bash
git clone <repo-url> && cd LLaMA-OSS
pip install -r requirements.txt   # includes dvc
python -m dvc pull                  # needs remote configured + credentials
```

Without `pull`, `LLaMA-Factory/data`, `outputs`, `merged_grpo_data.jsonl`, and `train_grpo.jsonl` are missing until you restore them from cache/remote or regenerate.

## Configure a remote (required for `push` / team `pull`)

Pick one backend (examples):

```bash
# Local directory or NAS (path must exist)
python -m dvc remote add -d storage /mnt/shared/llama-oss-dvc

# Amazon S3 (set AWS creds separately)
python -m dvc remote add -d storage s3://your-bucket/llama-oss-dvc

# Google Cloud Storage
python -m dvc remote add -d storage gs://your-bucket/llama-oss-dvc
```

`storage` is the default name; change if you prefer. Then:

```bash
python -m dvc push
```

## After changing tracked data

```bash
python -m dvc add LLaMA-Factory/data outputs merged_grpo_data.jsonl train_grpo.jsonl
git add LLaMA-Factory/data.dvc outputs.dvc merged_grpo_data.dvc train_grpo.dvc .gitignore LLaMA-Factory/.gitignore
git commit -m "Update DVC datasets"
python -m dvc push
git push
```

## Repo settings

- `core.autostage` is enabled in `.dvc/config` so `dvc` can restage `.dvc` files automatically.
- Ignore rules: root `.gitignore` lists `/outputs`, `/merged_grpo_data.jsonl`, and `/train_grpo.jsonl`; `LLaMA-Factory/.gitignore` lists `/data` so raw trees are not recommitted to Git.
