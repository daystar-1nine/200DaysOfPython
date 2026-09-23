"""
Training callbacks module for Day 105: Neural NLP & Text Classification.
Provides EarlyStopping to mitigate overfitting.
"""

from typing import Optional


class EarlyStopping:
    """Monitors a validation metric and halts training when improvement ceases."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False

        if self.mode not in ("min", "max"):
            raise ValueError(f"Mode must be 'min' or 'max', got {self.mode}")

    def __call__(self, current_score: float) -> bool:
        """Update tracker with current metric value.
        
        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
