"""
Challenge 1: Implement BERT's Additive Input Embeddings in pure NumPy.
Formula: E_input = E_token + E_position + E_segment
Followed by Layer Normalization: y = gamma * (x - mu) / sqrt(sigma^2 + eps) + beta.
"""
from typing import Optional
import numpy as np


class BertAdditiveEmbedding:
    """
    Constructs BERT's additive input representations:
    Sums Token, Learned Position, and Segment embeddings followed by Layer Normalization.
    """
    def __init__(
        self,
        vocab_size: int = 1000,
        max_positions: int = 128,
        type_vocab_size: int = 2,
        hidden_dim: int = 64,
        eps: float = 1e-12,
        seed: int = 42
    ):
        rng = np.random.default_rng(seed)
        scale = 0.02
        self.word_embeddings = rng.normal(0, scale, (vocab_size, hidden_dim))
        self.position_embeddings = rng.normal(0, scale, (max_positions, hidden_dim))
        self.token_type_embeddings = rng.normal(0, scale, (type_vocab_size, hidden_dim))

        self.gamma = np.ones(hidden_dim, dtype=np.float32)
        self.beta = np.zeros(hidden_dim, dtype=np.float32)
        self.eps = eps

    def forward(
        self,
        input_ids: np.ndarray,
        token_type_ids: Optional[np.ndarray] = None,
        position_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        batch_size, seq_len = input_ids.shape
        if token_type_ids is None:
            token_type_ids = np.zeros((batch_size, seq_len), dtype=np.int64)
        if position_ids is None:
            position_ids = np.broadcast_to(np.arange(seq_len), (batch_size, seq_len))

        # Lookups
        tok_vec = self.word_embeddings[input_ids]
        pos_vec = self.position_embeddings[position_ids]
        seg_vec = self.token_type_embeddings[token_type_ids]

        # Additive combination
        total = tok_vec + pos_vec + seg_vec

        # Layer Normalization
        mean = np.mean(total, axis=-1, keepdims=True)
        var = np.var(total, axis=-1, keepdims=True)
        normed = self.gamma * ((total - mean) / np.sqrt(var + self.eps)) + self.beta
        return normed


if __name__ == "__main__":
    embedder = BertAdditiveEmbedding(vocab_size=500, max_positions=64, hidden_dim=32)
    inputs = np.array([[10, 45, 88], [2, 102, 0]])
    out = embedder.forward(inputs)
    print("Embedding output shape:", out.shape)
    assert out.shape == (2, 3, 32)
    assert np.allclose(np.mean(out, axis=-1), 0.0, atol=1e-5)
    print("Challenge 1 passed successfully!")
