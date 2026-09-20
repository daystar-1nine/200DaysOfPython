"""Negative sampling loss computation."""
import numpy as np

def compute_loss(pos_prob: float, neg_probs: np.ndarray, eps: float = 1e-12) -> float:
    """
    Computes Negative Sampling Skip-Gram loss:
    L = -log(pos_prob) - sum(log(1 - neg_probs))
    """
    pos_loss = -np.log(np.clip(pos_prob, eps, 1.0))
    neg_loss = -np.sum(np.log(np.clip(1.0 - neg_probs, eps, 1.0)))
    return float(pos_loss + neg_loss)
