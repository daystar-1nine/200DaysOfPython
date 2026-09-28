import torch
import torch.nn as nn
from typing import Optional, Tuple

class AttentionLayer(nn.Module):
    """
    Mask-Aware Attention Layer for sequential representations.
    Computes an attention score for each timestep in the sequence:
      score_t = v^T tanh(W_a h_t + b_a)
    If mask is provided (1 for valid token, 0 for PAD), padding tokens receive -1e9
    prior to softmax so that their attention weight is approximately 0.
    
    Returns:
      context: Weighted sum of representations (batch_size, hidden_dim)
      weights: Attention distribution across timesteps (batch_size, seq_len)
    """
    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        inputs: (batch_size, seq_len, hidden_dim)
        mask: Optional (batch_size, seq_len) bool or int tensor (1 = valid, 0 = pad)
        """
        # (batch_size, seq_len, 1)
        scores = self.projection(inputs)
        scores = scores.squeeze(-1) # (batch_size, seq_len)
        
        if mask is not None:
            # Mask out padding positions with large negative constant
            pad_mask = (mask == 0)
            scores = scores.masked_fill(pad_mask, -1e9)
            
        weights = torch.softmax(scores, dim=-1) # (batch_size, seq_len)
        
        # Compute context vector: (batch_size, 1, seq_len) @ (batch_size, seq_len, hidden_dim)
        # -> (batch_size, 1, hidden_dim) -> (batch_size, hidden_dim)
        weights_expanded = weights.unsqueeze(1)
        context = torch.bmm(weights_expanded, inputs).squeeze(1)
        
        return context, weights


# TensorFlow builder with mask propagation matching Section 19 & 22
def build_tf_attention_layer():
    try:
        import tensorflow as tf
        class TFAttentionLayer(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.score = tf.keras.layers.Dense(1)

            def call(self, inputs, mask=None):
                scores = self.score(inputs)
                if mask is not None:
                    scores = tf.where(
                        mask[..., None],
                        scores,
                        tf.constant(-1e9, dtype=scores.dtype)
                    )
                weights = tf.nn.softmax(scores, axis=1)
                context = tf.reduce_sum(inputs * weights, axis=1)
                return context
        return TFAttentionLayer
    except ImportError:
        return None
