# ViT Fine-Tuning on Food-101

Comparing different strategies for adapting a pretrained **Vision Transformer (ViT)** to Food-101 image classification.

## Setup

* **Dataset:** Food-101
* **Classes:** 101
* **Train:** 60,600 images
* **Validation:** 15,150 images
* **Model:** `google/vit-base-patch16-224`
* **Image Size:** 224 × 224
* **Training:** 5 epochs
* **Framework:** PyTorch, Hugging Face Transformers, PEFT

## Methods

### Linear Probing

The ViT backbone is frozen and only the classification head is trained.

* **Trainable Parameters:** 77,669
* **Trainable:** 0.0904%

### LoRA

LoRA adapters are applied to the query and value projections (`q_proj`, `v_proj`) of the ViT attention layers.

* **Rank:** 8
* **Alpha:** 16
* **Dropout:** 0.1
* **Trainable Parameters:** 372,581
* **Trainable:** 0.4320%

### Full Fine-Tuning

All ViT parameters are updated during training.

* **Trainable Parameters:** 85,876,325
* **Trainable:** 100%

## Results

| Method           | Trainable Params | Trainable % | Validation Accuracy |
| ---------------- | ---------------: | ----------: | ------------------: |
| Linear Probing   |           77,669 |     0.0904% |          **81.82%** |
| LoRA             |          372,581 |     0.4320% |          **85.35%** |
| Full Fine-Tuning |       85,876,325 |        100% |          **85.09%** |

LoRA achieved the highest validation accuracy while updating only **0.43%** of the model parameters. It improved accuracy by **3.53 percentage points** over linear probing and slightly outperformed full fine-tuning.

## Project Structure

```text
notebooks/
├── 01_linear_probing.ipynb
├── 02_lora_finetuning.ipynb
└── 03_full_finetuning.ipynb
```

## Tech Stack

`Python` · `PyTorch` · `Transformers` · `PEFT` · `Hugging Face Datasets` · `Torchvision`
