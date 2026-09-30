"""
Checkpointing utilities for saving and restoring training state:
model parameters, optimizer states, scheduler steps, and run metadata.
"""
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn


def save_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None
) -> None:
    """Saves complete training state dictionary to disk."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": config or {},
        "metrics": metrics or {}
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, filepath)


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Restores model weights, optimizer state, and scheduler step from disk checkpoint.
    Returns checkpoint metadata dictionary.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device or "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "step": checkpoint.get("step", 0),
        "config": checkpoint.get("config", {}),
        "metrics": checkpoint.get("metrics", {})
    }
