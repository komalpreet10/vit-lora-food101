# Efficient Fine-Tuning of ViT on Food-101

Comparing **LoRA** and **Linear Probing** for efficient adaptation of a pretrained Vision Transformer (ViT) on the Food-101 image classification dataset.

## Overview

- **Dataset:** Food-101
- **Classes:** 101
- **Model:** `google/vit-base-patch16-224`
- **Methods:** LoRA, Linear Probing
- **Framework:** PyTorch, Hugging Face Transformers, PEFT
- **Training:** 5 epochs
- **GPU:** NVIDIA T4

## LoRA

LoRA freezes the pretrained ViT weights and introduces small trainable low-rank matrices into selected attention layers:

\[
W' = W + BA
\]

LoRA was applied to the **query (`q_proj`) and value (`v_proj`) projections** of the self-attention layers.

### LoRA Configuration

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
```

- **Trainable parameters:** 372,581
- **Trainable percentage:** 0.43%

## Linear Probing

For linear probing, the entire pretrained ViT backbone is frozen and only the final 101-class classification head is trained.

- **Trainable parameters:** 77,669
- **Trainable percentage:** 0.09%

## Results

| Method | Trainable Params | Trainable % | Validation Accuracy |
|---|---:|---:|---:|
| Linear Probing | 77,669 | 0.09% | TBD |
| LoRA | 372,581 | 0.43% | **85.35%** |

### LoRA Training

| Epoch | Train Loss | Val Loss | Accuracy |
|---:|---:|---:|---:|
| 1 | 0.8345 | 0.6525 | 82.48% |
| 2 | 0.7218 | 0.5972 | 83.99% |
| 3 | 0.6201 | 0.5724 | 84.68% |
| 4 | 0.5604 | 0.5585 | 85.16% |
| 5 | 0.4950 | 0.5500 | **85.35%** |

## Tech Stack

`Python` · `PyTorch` · `Transformers` · `PEFT` · `Hugging Face Datasets` · `Torchvision`
