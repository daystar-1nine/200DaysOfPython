"""
Visualization module for Positional Encoding heatmaps.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def save_positional_encoding_heatmap(
    pe: np.ndarray,
    output_path: Path,
    max_len_display: int = 100
) -> None:
    """
    Renders and saves a high-resolution heatmap of the sinusoidal positional encodings.
    """
    pe_slice = pe[:max_len_display, :]
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.pcolormesh(pe_slice, cmap="viridis", shading="auto")
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Encoding Signal Value", rotation=270, labelpad=15)

    ax.set_xlabel("Embedding Dimension (i)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Sequence Position (pos)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Sinusoidal Positional Encoding (Max Len={pe_slice.shape[0]}, d_model={pe_slice.shape[1]})",
        fontsize=12,
        fontweight="bold"
    )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
