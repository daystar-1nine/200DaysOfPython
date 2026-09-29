"""
Challenge 5: Implement BERT Attention Mask & Additive Attention Bias Generator.
Transforms binary attention mask (1=real token, 0=padding) into additive bias matrix:
(1 - mask) * -10000.0, ensuring masked padding tokens receive zero softmax attention weight.
"""
from typing import Tuple
import numpy as np
import torch


def create_attention_bias(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Args:
        attention_mask: (batch_size, seq_len) with 1 for tokens, 0 for pad.
    Returns:
        attention_bias: (batch_size, 1, 1, seq_len) with 0.0 for tokens, -10000.0 for pad.
    """
    extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    attention_bias = (1.0 - extended_mask.float()) * -10000.0
    return attention_bias


if __name__ == "__main__":
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.long)
    bias = create_attention_bias(mask)
    print("Attention bias shape:", bias.shape)
    assert bias.shape == (2, 1, 1, 5)
    assert bias[0, 0, 0, 0].item() == 0.0
    assert bias[0, 0, 0, 3].item() == -10000.0
    print("Challenge 5 passed successfully!")
