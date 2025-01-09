# Ranking-Loss-with-DeBERTa
A PyTorch implementation of a ranking model using DeBERTa-v3 with a custom ranking loss function. The project uses Hydra for configuration management and Accelerate for distributed training support.

# Features

1. DeBERTa-v3 base model with custom ranking head
2. Custom ranking loss implementation with margin
3. Layer-wise Learning Rate Decay (LLRD) optimization
4. Gradient checkpointing for memory efficiency
5. Distributed training support via Accelerate
6. Wandb integration for experiment tracking
7. Hydra configuration management
