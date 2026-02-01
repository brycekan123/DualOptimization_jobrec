# De-conflating Preference and Qualification: Constrained Dual-Perspective Reasoning for Job Recommendation with Large Language Models

This repository contains the implementation for training and evaluating LLM-based job recommendation models using multi-task learning and Lagrangian optimization.

## Requirements

- GPU: A40 or A100 recommended
- Python 3.8+
- See `requirements.txt` for dependencies

## Overview

The training pipeline consists of three stages:

1. **Stage 1A**: Train preference head only (single-task)
2. **Stage 1B**: Multi-task training for both preference and qualification heads
3. **Stage 2**: Train lambda head with Lagrangian optimization using Stage 1B checkpoint

## Usage

### Stage 1A: Preference-Only Training

Train the model on preference data only.

```bash
python3 stage_1A.py --dataset ../dataset --hf_token YOUR_HF_TOKEN_HERE
```

Evaluate on preference test set.

```bash
python3 eval_label_pref_full_test.py --hf_token YOUR_HF_TOKEN_HERE
```

Evaluate on qualification test set (cross-evaluation).

```bash
python3 eval_qual_test.py --hf_token YOUR_HF_TOKEN_HERE
```

**Note**: A pre-trained Stage 1A checkpoint is provided in the `12_17_stage1A` folder. This stage can be skipped if you want to proceed directly to Stage 1B.

### Baseline: Zero-Shot Evaluation

Evaluate vanilla Llama-3-8B-Instruct on preference ranking.

```bash
python3 label_pref_0_shot.py --hf_token YOUR_HF_TOKEN_HERE
```
Evaluate Llama-3-8B-Instruct with In-Context Learning (ICL) on preference ranking.
```bash
python3 label_pref_ICL.py --hf_token YOUR_HF_TOKEN_HERE
```
Evaluate vanilla Llama-3-8B-Instruct on qualification ranking.

```bash
python3 label_qual_0_shot.py --hf_token YOUR_HF_TOKEN_HERE
```
Evaluate Llama-3-8B-Instruct with In-Context Learning (ICL) on qualification ranking.

```bash
python3 label_qual_ICL.py --hf_token YOUR_HF_TOKEN_HERE
```
**Note**: This baseline evaluation is optional and can be skipped.

### Stage 1B: Multi-Task Training

Train both preference and qualification heads jointly with shared LoRA adapters.

```bash
python3 stage_1B.py --dataset ../dataset --hf_token YOUR_HF_TOKEN_HERE
```

### Stage 2: Lagrangian Policy Training

Train lambda head to balance preference and qualification using constrained optimization (requires Stage 1B checkpoint).

```bash
python tau_0.01.py \
  --dataset ../dataset \
  --stage1b_checkpoint ../stage_1B/stage1b_multitask_output/5e-5_head_LoRA/checkpoint_epoch3 \
  --hf_token YOUR_HF_TOKEN_HERE
```

**Note**: A pre-trained Stage 1B checkpoint is provided in the `stage_1B/stage1b_multitask_output/5e-5_head_LoRA/` folder.

## Citation

If you use this code, please cite our paper:

```bibtex
[Citation to be added]
```
