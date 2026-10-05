"""
Checkpointing utilities for saving and restoring SFT models and LoRA adapters.
"""
from pathlib import Path
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn


def save_sft_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    lora_only: bool = False
) -> None:
    """
    Saves SFT training checkpoint to disk.
    If lora_only is True, extracts and saves only the trainable LoRA adapter weights.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if lora_only:
        # Extract only parameters with requires_grad=True (LoRA parameters)
        state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    else:
        state_dict = model.state_dict()

    checkpoint_data = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": state_dict,
        "lora_only": lora_only,
        "config": config or {},
        "metrics": metrics or {}
    }

    if optimizer is not None:
        checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint_data, filepath)


def load_sft_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Restores model weights, optimizer state, and scheduler step from disk checkpoint.
    Handles both full model checkpoints and LoRA-only adapter checkpoints.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    target_device = device or torch.device("cpu")
    checkpoint = torch.load(filepath, map_location=target_device)

    is_lora_only = checkpoint.get("lora_only", False)
    if is_lora_only:
        # Load LoRA adapter weights into existing model
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", {}),
        "lora_only": is_lora_only
    }
