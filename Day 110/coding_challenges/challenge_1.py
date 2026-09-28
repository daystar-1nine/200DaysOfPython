"""
Challenge 1: Implement sinusoidal positional encoding using only NumPy.
"""
import numpy as np


def positional_encoding(max_length: int, d_model: int) -> np.ndarray:
    """
    Computes sinusoidal positional encoding matrix of shape (max_length, d_model).
    """
    pe = np.zeros((max_length, d_model), dtype=np.float32)
    positions = np.arange(max_length)[:, np.newaxis]
    dimensions = np.arange(d_model)[np.newaxis, :]

    angle_rates = 1.0 / np.power(10000.0, (2 * (dimensions // 2)) / d_model)
    angles = positions * angle_rates

    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


if __name__ == "__main__":
    pe = positional_encoding(50, 64)
    print("PE shape:", pe.shape)
    assert pe.shape == (50, 64)
    assert np.isfinite(pe).all()
    assert np.allclose(pe[0, 0], 0.0)
    assert np.allclose(pe[0, 1], 1.0)
    print("Challenge 1 passed!")
