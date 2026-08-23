# ViT Fine-Tuning with LoRA on Food-101

Parameter-efficient fine-tuning of a pretrained **Vision Transformer (ViT)** for food image classification using **Low-Rank Adaptation (LoRA)**.

## Overview

- **Dataset:** Food-101
- **Classes:** 101
- **Model:** `google/vit-base-patch16-224-in21k`
- **Method:** LoRA (PEFT)
- **Framework:** PyTorch, Hugging Face Transformers, PEFT
- **Training:** 5 epochs

## Results

| Epoch | Train Loss | Val Loss | Accuracy |
|---:|---:|---:|---:|
| 1 | 0.8345 | 0.6525 | 82.48% |
| 2 | 0.7218 | 0.5972 | 83.99% |
| 3 | 0.6201 | 0.5724 | 84.68% |
| 4 | 0.5604 | 0.5585 | 85.16% |
| 5 | 0.4950 | 0.5500 | **85.35%** |

**Final Validation Accuracy: 85.35%**

## Approach

LoRA keeps the pretrained ViT weights frozen and learns small low-rank adapter matrices:

\[
W' = W + BA
\]

This enables task adaptation while training only a small fraction of the model parameters.

## Baseline Comparison

| Method | Validation Accuracy |
|---|---:|
| Linear Probing | TBD |
| LoRA | **85.35%** |

## Tech Stack

`Python` · `PyTorch` · `Transformers` · `PEFT` · `Hugging Face Datasets`