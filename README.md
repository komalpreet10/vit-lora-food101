# Parameter-Efficient Fine-Tuning of ViT with LoRA

Fine-tuning a **Vision Transformer (ViT)** for Food-101 image classification using **Low-Rank Adaptation (LoRA)**. The goal is to evaluate whether parameter-efficient fine-tuning can achieve strong classification performance while updating only a small fraction of the model's parameters.

## Overview

Full fine-tuning updates every parameter of a pretrained model, which can be computationally expensive. **LoRA** provides a parameter-efficient alternative by freezing the pretrained weights and introducing small trainable low-rank matrices into selected layers.

This project applies LoRA to the self-attention layers of a pretrained ViT and fine-tunes it for **101-class food image classification**.

## Dataset

**Food-101** contains images belonging to **101 food categories**.

For this experiment, a subset of **5,000 images** was used and split into:

* **4,000 training images**
* **1,000 validation images**

Training images were augmented using random resized cropping and horizontal flipping. Validation images used deterministic resizing and center cropping.

## Model

**Base Model:** `google/vit-base-patch16-224`

The model is a ViT-Base architecture that processes `224 × 224` images using `16 × 16` patches.

The original ImageNet classification head was replaced with a new:

```text
Linear(768 → 101)
```

classification layer for Food-101.

## LoRA Configuration

LoRA adapters were added to the **query (`q_proj`) and value (`v_proj`) projections** of the ViT self-attention layers.

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"]
)
```

The pretrained ViT backbone remains frozen while the LoRA parameters and classification head are optimized.

### Parameter Efficiency

| Metric                   |     Value |
| ------------------------ | --------: |
| Total Parameters         |    86.25M |
| Trainable Parameters     |    372.6K |
| Trainable Parameters (%) | **0.43%** |

LoRA therefore reduces the number of parameters requiring gradient updates by more than **99%** compared with full fine-tuning.

## Training

Training was performed on an **NVIDIA T4 GPU** using Hugging Face Transformers and PEFT.

| Configuration        | Value               |
| -------------------- | ------------------- |
| Epochs               | 5                   |
| Image Size           | 224 × 224           |
| Mixed Precision      | FP16                |
| Evaluation Metric    | Accuracy            |
| Best Model Selection | Validation Accuracy |

## Results

LoRA achieved a **best validation accuracy of 95.9%** while training only **0.43% of the model parameters**.

| Epoch | Training Loss | Validation Loss |  Accuracy |
| ----: | ------------: | --------------: | --------: |
|     1 |         0.280 |           0.172 |     93.7% |
|     2 |         0.198 |           0.149 |     94.8% |
|     3 |         0.153 |           0.139 |     95.6% |
|     4 |         0.150 |           0.132 | **95.9%** |
|     5 |         0.142 |           0.130 |     95.5% |

The results demonstrate that LoRA can effectively adapt a pretrained Vision Transformer to a downstream image-classification task while modifying less than **1% of the model parameters**.

## Tech Stack

**Python · PyTorch · Hugging Face Transformers · PEFT · Datasets · Torchvision · NumPy**

## Next Steps

The next stage of the project will benchmark LoRA against:

* **Linear probing** — freeze the ViT backbone and train only the classifier.
* **Full fine-tuning** — update all ViT parameters.
* **LoRA** — update only low-rank adapters and the classifier.

The comparison will focus on **validation accuracy, trainable parameters, training time, and GPU memory usage** to quantify the performance-efficiency trade-off between the three approaches.
