import torch
import torch.nn as nn

class BiGRUClassifier(nn.Module):
    """
    Bidirectional GRU Classifier matching Section 25:
      Embedding(vocab_size, embedding_dim=64)
      Bidirectional(GRU(64)) -> Hidden dim becomes 64 * 2 = 128
      Dropout(0.3)
      Dense(128 -> 32, ReLU)
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
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.bigru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim * 2, dense_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        # h_n shape: (num_directions * num_layers, batch_size, hidden_dim) -> (2, B, H)
        _, h_n = self.bigru(embedded)
        # Concatenate forward and backward final hidden states
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_cat = torch.cat([h_forward, h_backward], dim=-1) # (B, 2*H)
        
        out = self.dropout(h_cat)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
