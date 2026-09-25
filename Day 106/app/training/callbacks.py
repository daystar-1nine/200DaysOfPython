"""
Training callbacks for Day 106: RNNs & Sequential Text Learning.
Provides EarlyStopping and ModelCheckpoint mechanisms.
"""

from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn


class EarlyStopping:
    """Monitors validation loss and signals when training should terminate."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Update state and return True if training should stop.
        
        Args:
            val_loss: Current validation loss.
            model: Active PyTorch model.
            
        Returns:
            Boolean indicating if training should early stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def restore_best_weights(self, model: nn.Module) -> None:
        """Restore model weights from the best monitored epoch."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class ModelCheckpoint:
    """Saves best model weights to disk."""

    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self.best_loss = float("inf")

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Check if current val_loss is an improvement and save model if so."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.filepath)
            return True
        return False

    def load(self, model: nn.Module) -> nn.Module:
        """Load weights from checkpoint file into model."""
        if self.filepath.exists():
            model.load_state_dict(torch.load(self.filepath, map_location="cpu"))
        return model
