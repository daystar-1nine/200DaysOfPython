"""
Model checkpointing module for Day 105: Neural NLP & Text Classification.
Saves and reloads the best model weights based on validation performance.
"""

from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn


class ModelCheckpoint:
    """Saves model weights when the monitored metric reaches a new optimal value."""

    def __init__(
        self,
        filepath: Union[str, Path],
        mode: str = "min"
    ):
        self.filepath = Path(filepath)
        self.mode = mode.lower()
        self.best_metric: Optional[float] = None

        if self.mode not in ("min", "max"):
            raise ValueError(f"Mode must be 'min' or 'max', got {self.mode}")

    def step(self, current_metric: float, model: nn.Module) -> bool:
        """Check if current metric is the best seen so far, and save weights if true.
        
        Args:
            current_metric: Monitored metric value.
            model: PyTorch model instance.
            
        Returns:
            True if model was saved, False otherwise.
        """
        is_best = False
        if self.best_metric is None:
            is_best = True
        elif self.mode == "min" and current_metric < self.best_metric:
            is_best = True
        elif self.mode == "max" and current_metric > self.best_metric:
            is_best = True

        if is_best:
            self.best_metric = current_metric
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.filepath)
            return True

        return False

    def load_best(self, model: nn.Module) -> nn.Module:
        """Load the saved best weights into the provided model instance."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.filepath}")
        model.load_state_dict(torch.load(self.filepath))
        return model
