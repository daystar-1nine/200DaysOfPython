"""
Challenge 8: Implement Transfer Learning Parameter Freezer.
Freezes or unfreezes backbone parameters and returns parameter statistics.
"""
from typing import Dict, Any
import torch
import torch.nn as nn


class MockTransformer(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def freeze_encoder(self, freeze: bool = True) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = not freeze

    def parameter_summary(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}


if __name__ == "__main__":
    model = MockTransformer(hidden_dim=32)
    # 1. Initially unfrozen
    stats_unfrozen = model.parameter_summary()
    assert stats_unfrozen["frozen"] == 0
    assert stats_unfrozen["trainable"] == stats_unfrozen["total"]

    # 2. Freeze encoder
    model.freeze_encoder(True)
    stats_frozen = model.parameter_summary()
    assert stats_frozen["frozen"] > 0
    assert stats_frozen["trainable"] == sum(p.numel() for p in model.head.parameters())

    # 3. Unfreeze
    model.freeze_encoder(False)
    stats_restored = model.parameter_summary()
    assert stats_restored["frozen"] == 0

    print("Parameter summary frozen:", stats_frozen)
    print("Parameter summary unfrozen:", stats_restored)
    print("Challenge 8 passed successfully!")
