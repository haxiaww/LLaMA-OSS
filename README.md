ssh-keygen -t ed25519 -C "devkoi2k3@gmail.com"
cat ~/.ssh/id_ed25519.pub
ssh -T git@github.com
pip install uv
uv pip install -e ".[torch,metrics]" --no-build-isolation
uv pip install vllm