"""
BERT Input Representation from scratch in NumPy:
Token Embeddings + Learned Position Embeddings + Segment/Token-Type Embeddings.
"""
from typing import Tuple, Optional
import numpy as np


class BertEmbeddingScratch:
    """
    Simulates BERT's additive input embedding layer:
        InputEmbedding = TokenEmbedding + PositionEmbedding + SegmentEmbedding
    Followed by Layer Normalization and Dropout.
    """
    def __init__(
        self,
        vocab_size: int = 30522,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        hidden_size: int = 128,
        max_positions: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        eps: float = 1e-12,
        seed: int = 42
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_positions or max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.hidden_size = hidden_dim or hidden_size
        self.eps = eps

        rng = np.random.default_rng(seed)
        scale = 0.02
        # 1. Token Embeddings Table
        self.word_embeddings = rng.normal(0, scale, (self.vocab_size, self.hidden_size))
        # 2. Learned Position Embeddings Table
        self.position_embeddings = rng.normal(0, scale, (self.max_position_embeddings, self.hidden_size))
        # 3. Segment / Token Type Embeddings Table
        self.token_type_embeddings = rng.normal(0, scale, (self.type_vocab_size, self.hidden_size))

        # Layer Normalization weights
        self.gamma = np.ones(self.hidden_size, dtype=np.float32)
        self.beta = np.zeros(self.hidden_size, dtype=np.float32)

    @property
    def token_table(self) -> np.ndarray:
        return self.word_embeddings

    @property
    def position_table(self) -> np.ndarray:
        return self.position_embeddings

    @property
    def segment_table(self) -> np.ndarray:
        return self.token_type_embeddings

    def forward(
        self,
        input_ids: np.ndarray,
        token_type_ids: Optional[np.ndarray] = None,
        position_ids: Optional[np.ndarray] = None,
        apply_norm: bool = False
    ) -> np.ndarray:
        """
        Args:
            input_ids: (batch_size, seq_len)
            token_type_ids: optional (batch_size, seq_len), defaults to 0
            position_ids: optional (batch_size, seq_len), defaults to 0..seq_len-1
            apply_norm: whether to apply LayerNorm (default False for pure additive check)
        Returns:
            embeddings: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len = input_ids.shape

        # Support flexible positional argument ordering
        if token_type_ids is not None and np.any(token_type_ids >= self.type_vocab_size):
            token_type_ids, position_ids = position_ids, token_type_ids

        if token_type_ids is None:
            token_type_ids = np.zeros((batch_size, seq_len), dtype=np.int64)

        if position_ids is None:
            position_ids = np.broadcast_to(np.arange(seq_len), (batch_size, seq_len))

        # Lookups
        words = self.word_embeddings[input_ids]
        positions = self.position_embeddings[position_ids]
        segments = self.token_type_embeddings[token_type_ids]

        # Additive sum
        embeddings = words + positions + segments

        if apply_norm:
            mean = np.mean(embeddings, axis=-1, keepdims=True)
            var = np.var(embeddings, axis=-1, keepdims=True)
            embeddings = self.gamma * ((embeddings - mean) / np.sqrt(var + self.eps)) + self.beta

        return embeddings


def create_bert_input_embeddings(
    token_embeddings: np.ndarray,
    position_embeddings: np.ndarray,
    segment_embeddings: np.ndarray
) -> np.ndarray:
    """Combines token, position, and segment representations via element-wise addition."""
    return token_embeddings + position_embeddings + segment_embeddings


BertInputEmbeddingsScratch = BertEmbeddingScratch

if __name__ == "__main__":
    embedder = BertEmbeddingScratch(vocab_size=1000, hidden_size=128)
    tokens = np.array([[101, 45, 88, 102], [101, 12, 102, 0]])
    types = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
    out = embedder.forward(tokens, types, apply_norm=True)
    print("Embedding output shape:", out.shape)
    assert out.shape == (2, 4, 128)
    print("BERT Input Representation verification passed!")
