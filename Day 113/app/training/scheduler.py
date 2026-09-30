"""
Learning rate schedules for LLM pretraining:
- Cosine decay with linear warmup
- Linear decay with linear warmup
- Constant learning rate
"""
import math
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.10
) -> LambdaLR:
    """
    Creates learning rate scheduler with linear warmup followed by cosine decay down to min_lr_ratio.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0
) -> LambdaLR:
    """
    Creates learning rate scheduler with linear warmup followed by linear decay.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress)

    return LambdaLR(optimizer, lr_lambda)


def get_constant_schedule(optimizer: torch.optim.Optimizer) -> LambdaLR:
    """Creates a constant learning rate schedule (always multiplier 1.0)."""
    return LambdaLR(optimizer, lambda step: 1.0)


def configure_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    warmup_steps: int = 20,
    max_steps: int = 200,
    min_lr_ratio: float = 0.10
) -> LambdaLR:
    """Factory helper to instantiate desired learning rate schedule."""
    stype = scheduler_type.lower()
    if stype == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps, min_lr_ratio)
    elif stype == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, max_steps, min_lr_ratio)
    elif stype == "constant":
        return get_constant_schedule(optimizer)
    else:
        raise ValueError(f"Unknown scheduler type: '{scheduler_type}'. Expected 'cosine', 'linear', or 'constant'.")
