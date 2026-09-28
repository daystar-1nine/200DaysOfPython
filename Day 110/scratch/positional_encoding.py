"""
Sinusoidal Positional Encoding implemented from scratch in NumPy.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def positional_encoding(max_length: int, d_model: int) -> np.ndarray:
    """
    Computes sinusoidal positional encoding table of shape (max_length, d_model).

    Formula:
        PE(pos, 2i)   = sin(pos / (10000 ** (2i / d_model)))
        PE(pos, 2i+1) = cos(pos / (10000 ** (2i / d_model)))
    """
    if max_length <= 0 or d_model <= 0:
        raise ValueError("max_length and d_model must be positive integers.")

    pe = np.zeros((max_length, d_model), dtype=np.float32)

    positions = np.arange(max_length)[:, np.newaxis]  # (max_length, 1)
    dimensions = np.arange(d_model)[np.newaxis, :]    # (1, d_model)

    angle_rates = 1.0 / np.power(10000.0, (2 * (dimensions // 2)) / d_model)
    angles = positions * angle_rates  # (max_length, d_model)

    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    return pe


def plot_positional_encoding(pe: np.ndarray, save_path: Path = None) -> plt.Figure:
    """
    Generates a 2D heatmap of the sinusoidal positional encodings.
    Rows: Positions, Columns: Embedding Dimensions.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.pcolormesh(pe, cmap="viridis")
    fig.colorbar(cax, ax=ax, label="Encoding Value")
    ax.set_xlabel("Embedding Dimension Index (i)")
    ax.set_ylabel("Sequence Position (pos)")
    ax.set_title("Sinusoidal Positional Encoding Heatmap")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


if __name__ == "__main__":
    max_len = 100
    d_model = 128
    pe = positional_encoding(max_len, d_model)
    print(f"Generated PE shape: {pe.shape}")
    print(f"Position 0 even/odd values: sin={pe[0, 0]:.4f}, cos={pe[0, 1]:.4f}")
    assert pe.shape == (max_len, d_model)
    assert np.isfinite(pe).all()
    assert np.allclose(pe[0, 0], 0.0)
    assert np.allclose(pe[0, 1], 1.0)
    print("Positional encoding verification successful!")
