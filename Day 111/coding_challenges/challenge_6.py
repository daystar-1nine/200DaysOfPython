"""
Challenge 6: Implement BERT Classification Head (Pooler) in PyTorch.
Architecture: Dropout -> Linear(hidden_size, 64) -> ReLU -> Dropout -> Linear(64, num_classes) -> Sigmoid.
"""
import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int = 128, num_classes: int = 1, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, 64)
        self.act = nn.ReLU()
        self.out = nn.Linear(64, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, cls_repr: torch.Tensor) -> torch.Tensor:
        x = self.dropout(cls_repr)
        x = self.dense(x)
        x = self.act(x)
        x = self.dropout(x)
        logits = self.out(x)
        return self.sigmoid(logits)


if __name__ == "__main__":
    head = ClassificationHead(hidden_size=128, num_classes=1)
    cls_vectors = torch.randn(8, 128)
    probs = head(cls_vectors)
    print("Classification head probabilities shape:", probs.shape)
    assert probs.shape == (8, 1)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
    print("Challenge 6 passed successfully!")
