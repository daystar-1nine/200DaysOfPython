"""SGD optimizer with learning rate scheduler."""
import numpy as np

class SGDOptimizer:
    """Stochastic Gradient Descent optimizer with optional linear decay."""
    def __init__(self, lr: float = 0.025, min_lr: float = 0.0001):
        self.initial_lr = lr
        self.min_lr = min_lr
        self.current_lr = lr

    def update_learning_rate(self, current_step: int, total_steps: int):
        """Linearly decays learning rate from initial_lr to min_lr."""
        decay = (self.initial_lr - self.min_lr) * (current_step / max(1, total_steps))
        self.current_lr = max(self.min_lr, self.initial_lr - decay)

    def step(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Applies gradient descent update."""
        return param - self.current_lr * grad
