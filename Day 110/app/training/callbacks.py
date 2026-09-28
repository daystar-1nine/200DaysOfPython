"""
Training callbacks: EarlyStopping and ModelCheckpoint.
"""
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn


class EarlyStopping:
    """Early stops training when validation loss stops improving."""
    def __init__(self, patience: int = 4, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.early_stop: bool = False
        self.best_state = None

    def __call__(self, val_score: float, model: nn.Module) -> bool:
        score = -val_score if self.mode == "min" else val_score

        if self.best_score is None:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

        return False

    def restore(self, model: nn.Module) -> None:
        """Restores model weights to the best recorded checkpoint."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class ModelCheckpoint:
    """Saves model weights whenever monitored metric improves."""
    def __init__(self, filepath: Path, mode: str = "min"):
        self.filepath = Path(filepath)
        self.mode = mode
        self.best_score: Optional[float] = None

    def __call__(self, score: float, model: nn.Module) -> None:
        val = -score if self.mode == "min" else score
        if self.best_score is None or val > self.best_score:
            self.best_score = val
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.filepath)
