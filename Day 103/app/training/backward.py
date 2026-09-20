"""Analytical gradient calculation for Skip-Gram with Negative Sampling."""
import numpy as np

def compute_gradients(
    target_vec: np.ndarray,
    context_vec: np.ndarray,
    neg_vecs: np.ndarray,
    pos_prob: float,
    neg_probs: np.ndarray
) -> tuple:
    """
    Computes gradients of loss with respect to:
    - target vector (grad_target)
    - context vector (grad_context)
    - negative vectors (grad_neg)
    
    dL/dz_pos = pos_prob - 1
    dL/dz_neg = neg_probs
    """
    pos_err = pos_prob - 1.0  # Scalar
    neg_errs = neg_probs      # Shape: (K,)
    
    # Gradient w.r.t context vector: pos_err * target_vec
    grad_context = pos_err * target_vec
    
    # Gradient w.r.t negative vectors: neg_errs[:, None] * target_vec[None, :]
    grad_neg = neg_errs[:, np.newaxis] * target_vec[np.newaxis, :]
    
    # Gradient w.r.t target vector: pos_err * context_vec + sum(neg_errs * neg_vec)
    grad_target = pos_err * context_vec + np.sum(neg_errs[:, np.newaxis] * neg_vecs, axis=0)
    
    return grad_target, grad_context, grad_neg
