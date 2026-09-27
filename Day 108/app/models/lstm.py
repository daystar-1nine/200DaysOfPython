import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    Standard LSTM text classifier for benchmarking:
      Embedding(vocab_size, embedding_dim) -> LSTM(hidden_dim) -> Dropout(0.3) -> Dense(32) -> Dropout(0.2) -> Dense(1)
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
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, dense_dim)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(dense_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, (h_n, c_n) = self.lstm(embedded)
        h_last = h_n[-1]
        out = self.dropout1(h_last)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        return self.sigmoid(out).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
