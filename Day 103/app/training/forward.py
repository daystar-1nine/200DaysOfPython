"""Forward pass and activation functions."""
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function.
    Avoids overflow for large negative or positive z.
    """
    z = np.clip(z, -15.0, 15.0)
    return 1.0 / (1.0 + np.exp(-z))

def forward_step(
    target_vec: np.ndarray,
    context_vec: np.ndarray,
    neg_vecs: np.ndarray
) -> tuple:
    """
    Computes dot products and sigmoid probabilities for positive and negative pairs.
    Returns (pos_prob, neg_probs).
    """
    pos_dot = np.dot(target_vec, context_vec)
    pos_prob = float(sigmoid(pos_dot))
    
    neg_dots = np.dot(neg_vecs, target_vec)
    neg_probs = sigmoid(neg_dots)
    
    return pos_prob, neg_probs
