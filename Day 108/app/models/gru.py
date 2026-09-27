import torch
import torch.nn as nn
from typing import Optional

class GRUClassifier(nn.Module):
    """
    Standard PyTorch GRU SMS Spam Classifier matching Section 19 architecture:
      Embedding(vocab_size, embedding_dim=64, padding_idx=0)
      GRU(hidden_dim=64, batch_first=True)
      Dropout(dropout_rate=0.3)
      Dense(hidden_dim -> 32, ReLU)
      Dropout(0.2)
      Dense(32 -> 1, Sigmoid)
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        dense_dim: int = 32,
        dropout_rate: float = 0.3,
        pad_idx: int = 0
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, dense_dim)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(dense_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len)
        returns: (batch_size,) probabilities between 0 and 1
        """
        # (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # out: (batch_size, seq_len, hidden_dim), h_n: (1, batch_size, hidden_dim)
        _, h_n = self.gru(embedded)
        
        # Use final hidden state
        h_last = h_n[-1] # (batch_size, hidden_dim)
        
        out = self.dropout1(h_last)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        probs = self.sigmoid(out).squeeze(-1)
        return probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Optional TensorFlow builder matching Section 19
def build_tf_gru_model(vocab_size: int, max_length: int = 50):
    try:
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, mask_zero=True),
            tf.keras.layers.GRU(64),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        return model
    except ImportError:
        return None
