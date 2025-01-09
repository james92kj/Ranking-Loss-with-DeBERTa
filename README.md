# Ranking-Loss-with-DeBERTa

A PyTorch implementation of a ranking model using DeBERTa-v3 with a custom ranking loss function. The project uses Hydra for configuration management and Accelerate for distributed training support.

## Features

1. DeBERTa-v3 base model with custom ranking head
2. Custom ranking loss implementation with margin
3. Layer-wise Learning Rate Decay (LLRD) optimization
4. Gradient checkpointing for memory efficiency
5. Distributed training support via Accelerate
6. Wandb integration for experiment tracking
7. Hydra configuration management

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Key configurations in `cfg.yaml`:

```yaml
model:
  backbone_path: microsoft/deberta-v3-base
  gradient_checkpointing: true
  dropout_rate: 0.05
  max_length: 1024

train:
  gradient_accumulation_steps: 2
  per_device_train_batch_size: 64
  num_train_epochs: 3
  warmup_pct: 0.02
```

## Usage

Run training with multi-GPU support:
```bash
accelerate launch --multi-gpu --num_processes=2 --mixed-precision=fp16 main.py
```

## Requirements

Main dependencies:
```
transformers
hydra-core
accelerate
wandb
pandas
torch
bitsandbytes
```

See `requirements.txt` for the complete list.
