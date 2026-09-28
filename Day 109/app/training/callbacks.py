import copy
import torch.nn as nn

class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None

    def __call__(self, current_val: float, model: nn.Module) -> bool:
        score = -current_val if self.mode == "min" else current_val
        if self.best_score is None:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        return self.early_stop

    def restore(self, model: nn.Module):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
