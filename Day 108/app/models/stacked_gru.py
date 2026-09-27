import torch
import torch.nn as nn

class StackedGRUClassifier(nn.Module):
    """
    Stacked 2-Layer GRU Classifier matching Section 26:
      Embedding
        ↓
      GRU(hidden_dim=64, return_sequences=True)
        ↓
      Dropout(0.3)
        ↓
      GRU(hidden_dim=32, return_sequences=False)
        ↓
      Dense(32 -> 1)
        ↓
      Sigmoid
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        dropout_rate: float = 0.3,
        pad_idx: int = 0
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru1 = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim1,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.gru2 = nn.GRU(
            input_size=hidden_dim1,
            hidden_size=hidden_dim2,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        # gru1 returns all sequence steps (return_sequences=True equivalent)
        seq_out1, _ = self.gru1(embedded)
        dropped = self.dropout(seq_out1)
        # gru2 processes the sequence representations from gru1
        _, h_n2 = self.gru2(dropped)
        h_last = h_n2[-1]
        out = self.fc(h_last)
        return self.sigmoid(out).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
