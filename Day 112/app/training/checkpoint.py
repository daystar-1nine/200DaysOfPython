"""
Model state checkpointing and restoration utilities.
"""
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    filepath: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    loss: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Saves model state and training metadata to disk."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "loss": loss,
        "metadata": metadata or {}
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(state, filepath)


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Loads model state from disk and restores weights."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

    checkpoint = torch.load(filepath, map_location=device or "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
