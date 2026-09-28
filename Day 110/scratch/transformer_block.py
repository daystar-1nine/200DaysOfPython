"""
Transformer Encoder Block implemented from scratch in pure NumPy.
"""
from typing import Optional, Tuple
import numpy as np

try:
    from scratch.multi_head_attention import MultiHeadAttention
    from scratch.feed_forward import FeedForward
    from scratch.layer_norm import LayerNorm
except ImportError:
    from multi_head_attention import MultiHeadAttention
    from feed_forward import FeedForward
    from layer_norm import LayerNorm


class TransformerBlock:
    """
    Standard Transformer Encoder Block (Post-LN architecture):
        x1 = LayerNorm(x + MultiHeadAttention(x, x, x))
        x2 = LayerNorm(x1 + FeedForward(x1))
    """
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 256,
        activation: str = "relu",
        eps: float = 1e-5,
        seed: int = 42
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, seed=seed)
        self.ln1 = LayerNorm(d_model=d_model, eps=eps)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, activation=activation, seed=seed + 1)
        self.ln2 = LayerNorm(d_model=d_model, eps=eps)

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            x: Input array of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask (True at masked positions)
        Returns:
            output: Transformed representation of shape (batch_size, seq_len, d_model)
            attention_weights: Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        # Multi-Head Self-Attention sub-layer with residual connection & layer norm
        attn_out, weights = self.mha.forward(x, x, x, mask=mask)
        x_residual = x + attn_out
        x_norm1 = self.ln1.forward(x_residual)

        # Feed-forward sub-layer with residual connection & layer norm
        ffn_out = self.ffn.forward(x_norm1)
        x_final = self.ln2.forward(x_norm1 + ffn_out)

        return x_final, weights


if __name__ == "__main__":
    block = TransformerBlock(d_model=128, num_heads=4, d_ff=256)
    x = np.random.randn(2, 10, 128)
    out, weights = block.forward(x)
    print("TransformerBlock output shape:", out.shape)
    print("TransformerBlock attention weights shape:", weights.shape)
    assert out.shape == (2, 10, 128)
    assert weights.shape == (2, 4, 10, 10)
    print("NumPy TransformerBlock verification successful!")
