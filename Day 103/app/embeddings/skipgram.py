"""Single-step Skip-Gram execution."""
import numpy as np
from app.training.forward import forward_step
from app.training.loss import compute_loss
from app.training.backward import compute_gradients

def train_pair_step(
    target_id: int,
    context_id: int,
    neg_ids: list,
    W_in: np.ndarray,
    W_out: np.ndarray,
    lr: float
) -> float:
    """
    Executes forward pass, loss calculation, backpropagation, and parameter update
    for one (target, context) pair with negative samples.
    """
    target_vec = W_in[target_id]
    context_vec = W_out[context_id]
    neg_vecs = W_out[neg_ids]
    
    # 1. Forward pass
    pos_prob, neg_probs = forward_step(target_vec, context_vec, neg_vecs)
    
    # 2. Loss
    loss = compute_loss(pos_prob, neg_probs)
    
    # 3. Gradients
    grad_target, grad_context, grad_neg = compute_gradients(
        target_vec, context_vec, neg_vecs, pos_prob, neg_probs
    )
    
    # 4. In-place gradient descent updates
    W_in[target_id] -= lr * grad_target
    W_out[context_id] -= lr * grad_context
    for i, neg_id in enumerate(neg_ids):
        W_out[neg_id] -= lr * grad_neg[i]
        
    return loss
