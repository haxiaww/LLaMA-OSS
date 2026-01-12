# Setup Guide

## Prerequisites

### System Requirements
- **Python**: 3.8+ (Python 3.12+ recommended)
- **CUDA**: 11.8+ or 12.0+ (for GPU support)
- **RAM**: 16GB+ (32GB+ recommended for large models)
- **Storage**: 30GB+ free disk space
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2

### Hardware Requirements

#### Minimum Configuration
- **GPU**: NVIDIA RTX A5000 (24GB VRAM)
- **RAM**: 24GB
- **Storage**: 30GB+ free space

#### Recommended Configuration
- **GPU**: NVIDIA RTX A5000 (24GB VRAM)
- **RAM**: 32GB+
- **Storage**: 200GB+ SSD

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/Koii2k3/LLaMA-OSS.git
cd LlaMA-OSS
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Or using conda (recommended)
conda create -n vlai-gpt-oss python=3.12
conda activate vlai-gpt-oss
```

### 3. Install Core Dependencies
```bash
# Install framework
cd LLaMA-Factory && pip install -e . && cd ..
cd ms-swift && pip install -e . && cd ..

# Install other dependencies
pip install -r requirements.txt
```

## Environment Setup

### Environment Variables
```bash
# Add to ~/.bashrc or ~/.zshrc
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
export CUDA_VISIBLE_DEVICES="0"  # Specify GPU IDs
export TOKENIZERS_PARALLELISM=false

# HF token
export HF_TOKEN="hf_YOUR_ACCESS_TOKEN_HERE"
```

### CUDA Setup
```bash
# Verify CUDA installation
nvidia-smi
nvcc --version

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"
```

## Verification and Testing

### Quick Test Run
```bash
# Test inference with sample data
cd LLaMA-Factory/scripts
python 0_gpt.py \
  --model_name_or_path gpt-oss-20b \
  --dataset gsm8k \
  --save_name test.jsonl
```

### Permission Issues
```bash
# Fix file permissions
chmod +x scripts/*.py
chmod +x *.sh

# Fix directory permissions
chmod -R 755 LlaMA-Factory/
chmod -R 755 ms-swift/
```

## Next Steps

After successful setup:

1. **Read the [Usage Guide](USAGE.md)** for running inference and evaluation
2. **Check [Configuration Guide](CONFIG.md)** for customizing models and settings
3. **Review [Evaluation Guide](EVALUATION.md)** for benchmarking instructions
4. **Run sample experiments** to verify everything works correctly

## Support

If you encounter issues:

1. Review logs in `logs/` directory
2. Create an issue on GitHub with system info and error logs

---

**System Information Command:**
```bash
# Gather system info for support
python -c "
import torch, sys, platform
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"
```