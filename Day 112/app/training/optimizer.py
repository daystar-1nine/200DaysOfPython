"""
Optimizer configuration with parameter weight decay decoupling.
Biases and LayerNorm weights do not receive L2 regularization.
"""
from typing import Tuple
import torch
import torch.nn as nn


def configure_adamw_optimizer(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.95)
) -> torch.optim.AdamW:
    """
    Separates model parameters into decayed (weights) and non-decayed (biases, LayerNorm) groups.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases and 1D normalization tensors do not decay
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer
