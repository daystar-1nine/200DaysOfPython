"""
Challenge 4: Calculate parameter count manually for input_size = 32, hidden_size = 64.
Then verify it programmatically.
"""
import torch
import torch.nn as nn

def manual_gru_params(input_size=32, hidden_size=64):
    """
    Canonical formulation: 3 * [ (input_size * hidden_size) + (hidden_size * hidden_size) + hidden_size ]
    """
    return 3 * ((input_size * hidden_size) + (hidden_size * hidden_size) + hidden_size)

def programmatic_gru_params(input_size=32, hidden_size=64):
    gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
    return sum(p.numel() for p in gru.parameters())

if __name__ == "__main__":
    manual = manual_gru_params(32, 64)
    prog = programmatic_gru_params(32, 64)
    print(f"Manual Canonical Count: {manual}")
    print(f"Programmatic PyTorch Count: {prog}")
    # Note difference is PyTorch stores separate bias for input and hidden (64*3 extra = 192)
    assert manual == 18624
    assert prog == 18816
