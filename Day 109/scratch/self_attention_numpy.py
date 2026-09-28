"""
Self-Attention implementation from scratch in pure NumPy.
Includes linear projections Wq, Wk, Wv, scaled dot-product attention,
mask support, and attention matrix heatmap visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
try:
    from scratch.softmax_numpy import softmax
except ImportError:
    from softmax_numpy import softmax

class SelfAttention:
    """
    Pure NumPy implementation of Self-Attention with projection matrices.
    """
    def __init__(self, input_dim: int, attention_dim: int, seed: int = 42):
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        
        # Xavier/Glorot initialization for projections
        np.random.seed(seed)
        limit = np.sqrt(6.0 / (input_dim + attention_dim))
        self.Wq = np.random.uniform(-limit, limit, (input_dim, attention_dim))
        self.Wk = np.random.uniform(-limit, limit, (input_dim, attention_dim))
        self.Wv = np.random.uniform(-limit, limit, (input_dim, attention_dim))

    def forward(self, X: np.ndarray, mask: np.ndarray = None):
        """
        X: Input sequence of shape (batch_size, seq_len, input_dim) or (seq_len, input_dim)
        mask: Optional binary mask (1 for valid tokens, 0 for pad)
        
        Returns:
          context: Attended representations of shape (..., seq_len, attention_dim)
          weights: Attention weight matrix of shape (..., seq_len, seq_len)
        """
        is_2d = (X.ndim == 2)
        if is_2d:
            X = np.expand_dims(X, 0)
            if mask is not None and mask.ndim == 1:
                mask = np.expand_dims(mask, 0)
                
        # 1. Linear projections
        Q = X @ self.Wq # (batch, seq_len, attention_dim)
        K = X @ self.Wk # (batch, seq_len, attention_dim)
        V = X @ self.Wv # (batch, seq_len, attention_dim)
        
        # 2. Attention scores
        d_k = self.attention_dim
        # Q @ K^T: (batch, seq_len, seq_len)
        scores = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(d_k)
        
        # 3. Masking
        if mask is not None:
            # mask shape: (batch, seq_len) -> expand to (batch, 1, seq_len)
            mask_expanded = np.expand_dims(mask, 1)
            scores = np.where(mask_expanded > 0, scores, -1e9)
            
        # 4. Softmax weights
        weights = softmax(scores, axis=-1)
        
        # 5. Weighted context
        context = weights @ V # (batch, seq_len, attention_dim)
        
        if is_2d:
            context = np.squeeze(context, 0)
            weights = np.squeeze(weights, 0)
            
        return context, weights

def plot_attention_matrix(weights: np.ndarray, tokens: list, save_path: Path = None):
    """
    Plots attention matrix heatmap for a sequence of tokens.
    """
    plt.figure(figsize=(7, 6))
    plt.imshow(weights, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Attention Weight")
    
    plt.xticks(ticks=range(len(tokens)), labels=tokens, rotation=45, ha="right", fontsize=9)
    plt.yticks(ticks=range(len(tokens)), labels=tokens, fontsize=9)
    plt.xlabel("Key Tokens (Attended to)", fontsize=10, fontweight="bold")
    plt.ylabel("Query Tokens (Attending from)", fontsize=10, fontweight="bold")
    plt.title("Self-Attention Weight Matrix", fontsize=11, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.close()

if __name__ == "__main__":
    seq_len = 6
    in_dim = 16
    att_dim = 32
    
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    np.random.seed(42)
    X = np.random.randn(seq_len, in_dim)
    
    layer = SelfAttention(input_dim=in_dim, attention_dim=att_dim)
    context, weights = layer.forward(X)
    
    print("=== Self-Attention Scratch Validation ===")
    print(f"Input shape:   {X.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Weights shape: {weights.shape}")
    
    assert context.shape == (seq_len, att_dim)
    assert weights.shape == (seq_len, seq_len)
    assert np.allclose(weights.sum(axis=-1), 1.0)
    print("All shapes and probability sums verified!")
    
    # Save a demo plot
    out_dir = Path(__file__).resolve().parent.parent / "output" / "charts"
    plot_attention_matrix(weights, tokens, out_dir / "scratch_self_attention_matrix.png")
    print("Demo attention matrix saved.")
