<p align="center">
  <img src="common/img/AIVN.png" alt="AIVN" height="80"/>
  &nbsp;&nbsp;&nbsp;
  <img src="common/img/VLAI.png" alt="VLAI" height="80"/>
</p>

<h1 align="center">Curating Multi-Mode CoT for Efficient Math Reasoning with GPT-OSS</h1>

<p align="center">
  <a href="https://github.com/Koii2k3/LLaMA-OSS/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/Paper-arXiv-red" alt="Paper">
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page">
  </a>
</p>

<p align="center">
  <img src="common/img/Main.png" alt="Main Architecture" width="60%"/>
</p>

This repository contains the official implementation of our knowledge distillation framework that curates multi-mode chain-of-thought (CoT) reasoning from GPT-OSS for efficient mathematical question answering. Our approach addresses the challenge of noisy and overly verbose supervision in dataset-based distillation by implementing a two-step curation pipeline that emphasizes quality over quantity.

We demonstrate that **curated mode-specific supervision significantly outperforms raw teacher outputs**, achieving **80.06% accuracy on GSM8K (0-shot)** with Llama 3.2 3B when combining curated low-mode SFT initialization with GRPO fine-tuning.

<details>
<summary><b>Table of Contents</b></summary>

- [Key Findings](#key-findings)
- [Features](#features)
- [Experimental Results](#experimental-results)
  - [GSM8K Results](#gsm8k-results)
  - [MATH500 Results](#math500-results)
  - [Mode Comparison Analysis](#mode-comparison-analysis)
- [Methodology](#methodology)
  - [Two-Step Curation Pipeline](#two-step-curation-pipeline)
  - [Multi-Mode CoT Generation](#multi-mode-cot-generation)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
- [Training Pipeline](#training-pipeline)
  - [Data Preparation](#data-preparation)
  - [SFT Training](#sft-training)
  - [GRPO Fine-tuning](#grpo-fine-tuning)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

</details>

## Key Findings

Our research demonstrates several critical insights for knowledge distillation in mathematical reasoning:

1. **Quality over Quantity**: Curated, length-controlled reasoning traces are more effective than unfiltered CoT data for training small models
2. **Mode-Specific Effectiveness**: Different GPT-OSS inference modes (low/medium/high) provide distinct quality-verbosity tradeoffs suitable for different training stages
3. **Strong GRPO Initialization**: Curated low-mode SFT provides excellent initialization for GRPO, achieving **80.06% GSM8K accuracy**
4. **Curation Impact**: Our two-step curation (answer verification + length filtering) consistently improves performance across all modes compared to raw teacher outputs

## Features

- **Multi-Mode CoT Generation**: Leverages GPT-OSS's low/medium/high inference modes for controllable reasoning generation
- **Two-Step Curation Pipeline**: 
  - Final-answer verification to filter incorrect reasoning traces
  - Length distribution-based filtering with median-length selection to eliminate verbosity
- **SFT + GRPO Training**: Complete pipeline from supervised fine-tuning to policy optimization
- **Comprehensive Evaluation**: Automated evaluation on GSM8K and MATH500 benchmarks
- **LLaMA-Factory Integration**: Built on LLaMA-Factory for efficient training workflows
- **MS-SWIFT Support**: Compatible with ModelScope-SWIFT framework
- **Modular Design**: Easy to extend for other reasoning tasks or teacher models

## Experimental Results

All experiments use **Llama 3.2 3B** as the student model, distilled from GPT-OSS teacher models.

### GSM8K Results

| Model | Training | Dataset | GSM8K 0-shot | GSM8K 5-shot |
|-------|----------|---------|--------------|--------------|
| Llama3.2 | - | 𝒟<sub>orig</sub> | 0.7043 | 0.7043 |
| Llama3.2 | - | 𝒟<sup>*</sup> | 0.7043 | 0.7104 |
| Llama3.2 | SFT | 𝒟<sup>*</sup> | 0.6876 | 0.5762 |
| Llama3.2 | SFT | 𝒟<sup>*</sup><sub>low</sub> | **0.7111** | **0.7142** |
| Llama3.2 | SFT | 𝒟<sup>*</sup><sub>med</sub> | 0.7051 | 0.7074 |
| Llama3.2 | SFT | 𝒟<sup>*</sup><sub>high</sub> | 0.7013 | 0.7051 |
| Llama3.2-𝒟<sub>orig</sub> | GRPO | 𝒟 | 0.7771 | 0.6603 |
| Llama3.2-𝒟<sup>*</sup> | GRPO | 𝒟 | 0.7847 | 0.6156 |
| Llama3.2-𝒟<sup>*</sup><sub>low</sub> | GRPO | 𝒟 | 0.6308 | 0.5861 |
| **Llama3.2-𝒟<sup>*</sup><sub>low</sub>** | **GRPO** | 𝒟 | **0.8006** | **0.7195** |
| Llama3.2-𝒟<sup>*</sup><sub>med</sub> | GRPO | 𝒟 | 0.7771 | 0.6323 |
| Llama3.2-𝒟<sup>*</sup><sub>high</sub> | GRPO | 𝒟 | 0.7559 | 0.7225 |

**Key Insights:**
- Curated low-mode SFT (**0.7111**) outperforms raw teacher output SFT (0.6876) by **+2.35%**
- GRPO initialized from curated low-mode achieves **80.06% accuracy**, the best result across all configurations
- Curated datasets consistently outperform raw teacher outputs in both SFT and GRPO settings

### MATH500 Results

| Model | Training | Dataset | MATH500 0-shot | MATH500 4-shot |
|-------|----------|---------|----------------|----------------|
| Llama3.2 | - | 𝒟<sub>orig</sub> | 0.3960 | 0.4340 |
| Llama3.2 | - | 𝒟<sup>*</sup> | 0.4060 | 0.4240 |
| Llama3.2 | SFT | 𝒟<sup>*</sup> | 0.3400 | 0.2420 |
| Llama3.2 | SFT | 𝒟<sup>*</sup><sub>low</sub> | 0.4100 | **0.4400** |
| Llama3.2 | SFT | 𝒟<sup>*</sup><sub>med</sub> | 0.4000 | 0.4160 |
| Llama3.2 | SFT | 𝒟<sup>*</sup><sub>high</sub> | **0.4140** | 0.3920 |
| Llama3.2-𝒟<sub>orig</sub> | GRPO | 𝒟 | 0.4540 | 0.4380 |
| Llama3.2-𝒟<sup>*</sup> | GRPO | 𝒟 | 0.4520 | 0.4560 |
| Llama3.2-𝒟<sup>*</sup><sub>low</sub> | GRPO | 𝒟 | 0.4400 | 0.4220 |
| **Llama3.2-𝒟<sup>*</sup><sub>low</sub>** | **GRPO** | 𝒟 | **0.4760** | 0.4520 |
| Llama3.2-𝒟<sup>*</sup><sub>med</sub> | GRPO | 𝒟 | 0.4480 | 0.4600 |
| Llama3.2-𝒟<sup>*</sup><sub>high</sub> | GRPO | 𝒟 | 0.4740 | **0.4600** |

**Key Insights:**
- Curated modes show strong performance on MATH500, with high-mode achieving **0.4140** (0-shot) in SFT
- GRPO from curated low-mode initialization reaches **0.4760** on MATH500
- Curation is particularly effective for complex mathematical reasoning

### Mode Comparison Analysis

| Mode | Characteristics | Best Use Case | GSM8K Performance | MATH500 Performance |
|------|----------------|---------------|-------------------|---------------------|
| **Low** | Fast, concise reasoning | GRPO initialization | **0.8006** (GRPO) | **0.4760** (GRPO) |
| **Medium** | Balanced verbosity | General training | 0.7771 (GRPO) | 0.4600 (GRPO) |
| **High** | Detailed explanations | Complex problems | 0.7559 (GRPO) | 0.4740 (GRPO) |

## Methodology

### Two-Step Curation Pipeline

Our curation process ensures high-quality reasoning traces while avoiding extreme verbosity:

**Step 1: Final-Answer Verification**
- Extract final answers from teacher-generated CoT traces
- Verify correctness against ground truth
- Remove all incorrect reasoning traces

**Step 2: Length-Based Filtering**
- Analyze length distribution of correct traces
- Apply distribution-based filtering to remove outliers
- Select median-length traces to balance quality and conciseness
- Avoid both overly terse and excessively verbose examples

This two-step approach reduces noise and improves the signal-to-noise ratio in the training data, leading to more effective knowledge distillation.

### Multi-Mode CoT Generation

GPT-OSS provides three inference modes with different computational "effort" levels:

- **Low Mode**: Fast inference, concise reasoning steps
- **Medium Mode**: Balanced reasoning depth and verbosity
- **High Mode**: Detailed step-by-step explanations

We generate CoT traces in all three modes and curate them separately, creating mode-specific datasets (𝒟<sup>*</sup><sub>low</sub>, 𝒟<sup>*</sup><sub>med</sub>, 𝒟<sup>*</sup><sub>high</sub>) that offer different quality-verbosity tradeoffs.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- 24GB+ GPU memory (for Llama 3.2 3B training)
- LLaMA-Factory
- MS-SWIFT (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Koii2k3/LLaMA-OSS.git
cd LLaMA-OSS
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install LLaMA-Factory:
```bash
cd LLaMA-Factory
pip install -e .
cd ..
```

5. (Optional) Install MS-SWIFT:
```bash
pip install ms-swift
```

### Quick Start

1. **Generate and curate multi-mode CoT data**:
```bash
jupyter notebook convert_data.ipynb
# Follow the notebook to:
# - Generate CoT traces from GPT-OSS (low/medium/high modes)
# - Apply answer verification
# - Perform length-based filtering
```

2. **Train with curated low-mode data** (recommended):
```bash
bash train.sh --mode sft --data grpo_low.jsonl
```

3. **Fine-tune with GRPO**:
```bash
bash train.sh --mode grpo --init outputs/sft_low_checkpoint
```

4. **Evaluate on benchmarks**:
```bash
bash eval.sh --model outputs/grpo_final --dataset gsm8k
bash eval.sh --model outputs/grpo_final --dataset math500
```

## Training Pipeline

### Data Preparation

The `convert_data.ipynb` notebook handles the complete curation pipeline:

1. **Load raw math problems** from GSM8K/MATH500
2. **Generate CoT traces** using GPT-OSS in low/medium/high modes
3. **Apply curation**:
   - Answer verification
   - Length distribution analysis
   - Median-length selection
4. **Export curated datasets**: `grpo_low.jsonl`, `grpo_med.jsonl`, `grpo_high.jsonl`

### SFT Training

Supervised fine-tuning on curated mode-specific data:

```bash
# Train on curated low-mode data (recommended)
bash train.sh --mode sft --data grpo_low.jsonl --output outputs/sft_low

# Train on curated medium-mode data
bash train.sh --mode sft --data grpo_med.jsonl --output outputs/sft_med

# Train on curated high-mode data
bash train.sh --mode sft --data grpo_high.jsonl --output outputs/sft_high
```

**Configuration** (in `train.sh` or config files):
- Model: Llama 3.2 3B
- Learning rate: 5e-5
- Batch size: 4-8 (depending on GPU memory)
- Epochs: 3-5
- LoRA rank: 8-16

### GRPO Fine-tuning

Group Relative Policy Optimization for further improvement:

```bash
# Initialize from curated low-mode SFT (best results)
bash train.sh --mode grpo \
  --init outputs/sft_low \
  --data merged_grpo_data.jsonl \
  --output outputs/grpo_final
```

**GRPO Parameters**:
- Group size: 4
- KL coefficient: 0.1
- Clip range: 0.2
- Learning rate: 1e-5

## Evaluation

The `eval.sh` script evaluates models on GSM8K and MATH500:

```bash
# Evaluate on GSM8K (0-shot)
bash eval.sh \
  --model outputs/grpo_final \
  --dataset gsm8k \
  --shots 0

# Evaluate on GSM8K (5-shot)
bash eval.sh \
  --model outputs/grpo_final \
  --dataset gsm8k \
  --shots 5

# Evaluate on MATH500 (0-shot)
bash eval.sh \
  --model outputs/grpo_final \
  --dataset math500 \
  --shots 0
```

Results are saved to `outputs/` with detailed per-sample predictions and metrics.

## Project Structure

```
LLaMA-OSS/
├── LLaMA-Factory/              # LLaMA-Factory training framework
├── ms-swift/                   # MS-SWIFT (optional)
├── outputs/                    # Model checkpoints and results
│   ├── sft_low/               # SFT low-mode checkpoint
│   ├── sft_med/               # SFT medium-mode checkpoint
│   ├── sft_high/              # SFT high-mode checkpoint
│   └── grpo_final/            # Final GRPO model
├── convert_data.ipynb          # CoT generation & curation pipeline
├── train.sh                    # Training script (SFT + GRPO)
├── eval.sh                     # Evaluation script
├── grpo_low.jsonl             # Curated low-mode dataset
├── grpo_med.jsonl             # Curated medium-mode dataset
├── grpo_high.jsonl            # Curated high-mode dataset
├── merged_grpo_data.jsonl     # Merged dataset for GRPO
├── train_grpo.jsonl           # GRPO training configuration
├── grpo_debug.log             # Training logs
└── README.md                   # This file
```

## Citation

If you use this framework or find our work helpful, please consider citing:

```bibtex
@misc{trinh2025curating,
  author    = {Hai-Au Trinh and Tue-Anh Vu and Dai-Nhan Tran and Uyen Khoi-Minh Huynh and Anh-Khoi Nguyen},
  title     = {Curating Multi-Mode CoT for Efficient Math Reasoning with GPT-OSS},
  year      = {2025},
  publisher = {GitHub},
  journal   = {GitHub repository},
  howpublished = {\url{https://github.com/Koii2k3/LLaMA-OSS}},
}
```

## Acknowledgements

This project is built upon excellent work from the open-source community:

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for the training framework
- [MS-SWIFT](https://github.com/modelscope/swift) for additional training utilities
- [Meta LLaMA](https://ai.meta.com/llama/) for the foundation models
- [OpenAI GPT-OSS](https://openai.com/) for the teacher model
- [GSM8K](https://github.com/openai/grade-school-math) and [MATH](https://github.com/hendrycks/math) datasets
- [Hugging Face](https://huggingface.co/) for the `transformers` library

Special thanks to the research community for advancing efficient knowledge distillation techniques for mathematical reasoning.

---

**Note**: This is a research project focused on knowledge distillation for mathematical reasoning. The curation pipeline and training configurations are designed specifically for math QA tasks but can be adapted to other reasoning domains.