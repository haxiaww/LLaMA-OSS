pip install uv
uv pip install -e .
uv pip install lm-eval[math]
uv pip install vllm
pip install "datasets<4.0.0"
export HF_DATASETS_TRUST_REMOTE_CODE=1