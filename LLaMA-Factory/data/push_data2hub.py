#!/usr/bin/env python3
"""Push train/test JSONL datasets to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

from huggingface_hub import HfApi
try:
    from huggingface_hub import HfHubHTTPError
except ImportError:
    from huggingface_hub.utils import HfHubHTTPError

# load env
import dotenv
dotenv.load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a dataset repo on the Hub and push train/test jsonl files."
    )
    parser.add_argument("--repo-id", required=True, help="Dataset repository (e.g. user/my-dataset).")
    parser.add_argument("--train-file", default="data/combined_grpo_train.jsonl", help="Path to the train JSONL file.")
    parser.add_argument("--test-file", default="data/combined_val.jsonl", help="Path to the test JSONL file.")
    parser.add_argument(
        "--token",
        help="Hub token with permission to create/overwrite a dataset repo. Falls back to HF_TOKEN.",
        default=os.environ.get("HF_TOKEN"),
    )
    parser.add_argument(
        "--train-name",
        default="grpo_train.jsonl",
        help="Filename to use on the Hub for the train split.",
    )
    parser.add_argument(
        "--test-name",
        default="grpo_test.jsonl",
        help="Filename to use on the Hub for the test split.",
    )
    return parser.parse_args()


def resolve_token(cli_token: str | None) -> str:
    token = cli_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Your Hugging Face token must be provided via --token or HF_TOKEN.")
    return token


def validate_path(path: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{resolved} does not exist or is not a file.")
    return resolved


def ensure_repo(api: HfApi, repo_id: str, token: str) -> None:
    try:
        api.create_repo(repo_id=repo_id, token=token, repo_type="dataset")
    except HfHubHTTPError as exc:
        response = getattr(exc, "response", None)
        if response is not None and response.status_code == 409:
            return
        raise


def upload_split(api: HfApi, repo_id: str, local_path: Path, target_name: str, token: str) -> None:
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=target_name,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )


def main() -> None:
    args = parse_args()
    token = resolve_token(args.token)
    train_path = validate_path(args.train_file)
    test_path = validate_path(args.test_file)

    api = HfApi()
    ensure_repo(api, args.repo_id, token)

    upload_split(api, args.repo_id, train_path, args.train_name, token)
    upload_split(api, args.repo_id, test_path, args.test_name, token)

    print(
        f"Pushed train/test splits to https://huggingface.co/datasets/{args.repo_id} "
        f"with {args.train_name!r} and {args.test_name!r}."
    )


if __name__ == "__main__":
    main()
