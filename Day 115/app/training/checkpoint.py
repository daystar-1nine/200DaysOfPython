"""
Checkpointing utilities for Day 115: Preference Optimization & RLHF.
Handles saving and restoring Reward Models, DPO Policy models, and optimizer states.
"""
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import torch
import torch.nn as nn


def save_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Saves model and training state checkpoint to disk.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
        "config": config or {}
    }

    if optimizer is not None:
        data["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        data["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(data, filepath)
    return filepath


def load_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Restores model weights, optimizer, and scheduler states from disk.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    target_device = device or torch.device("cpu")
    data = torch.load(filepath, map_location=target_device)

    model.load_state_dict(data["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in data:
        optimizer.load_state_dict(data["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in data:
        scheduler.load_state_dict(data["scheduler_state_dict"])

    return {
        "epoch": data.get("epoch", 0),
        "step": data.get("step", 0),
        "metrics": data.get("metrics", {}),
        "config": data.get("config", {})
    }


class CheckpointManager:
    """
    Manages periodic checkpoint rotation, pruning old checkpoints to maintain max_keep.
    """
    def __init__(self, checkpoint_dir: Union[str, Path], max_keep: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        prefix: str = "ckpt"
    ) -> Path:
        filepath = self.checkpoint_dir / f"{prefix}_epoch_{epoch}_step_{step}.pt"
        save_checkpoint(filepath, model, optimizer, scheduler, epoch, step, metrics)
        self._prune_old_checkpoints(prefix)
        return filepath

    def load_checkpoint(
        self,
        filepath: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        return load_checkpoint(filepath, model, optimizer, scheduler, device=device, strict=strict)

    def list_checkpoints(self, prefix: str = "ckpt") -> List[Path]:
        ckpts = sorted(self.checkpoint_dir.glob(f"{prefix}_epoch_*.pt"), key=lambda p: p.stat().st_mtime)
        return ckpts

    def _prune_old_checkpoints(self, prefix: str = "ckpt") -> None:
        ckpts = self.list_checkpoints(prefix)
        if len(ckpts) > self.max_keep:
            to_remove = ckpts[:-self.max_keep]
            for p in to_remove:
                try:
                    p.unlink()
                except OSError:
                    pass

