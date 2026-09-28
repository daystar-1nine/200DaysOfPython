"""
Challenge 6: Implement create_padding_mask() and verify it visually.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def create_padding_mask(token_ids: np.ndarray, pad_id: int = 0) -> np.ndarray:
    """
    Creates boolean mask where True marks pad_id positions.
    """
    return np.asarray(token_ids) == pad_id


def plot_padding_mask(mask: np.ndarray, save_path: Path = None):
    fig, ax = plt.subplots(figsize=(6, 3))
    cax = ax.imshow(mask, cmap="binary", aspect="auto")
    fig.colorbar(cax, ax=ax, label="Mask (1 = Padded, 0 = Token)")
    ax.set_xlabel("Sequence Index")
    ax.set_ylabel("Batch Sample")
    ax.set_title("Padding Mask Visualization")
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    tokens = np.array([
        [15, 23, 8, 42, 0, 0, 0],
        [101, 84, 99, 12, 18, 5, 0]
    ])
    mask = create_padding_mask(tokens, pad_id=0)
    print("Mask:\n", mask.astype(int))
    assert mask.shape == (2, 7)
    assert mask[0, 4] and mask[0, 5] and mask[0, 6]
    assert not mask[0, 0] and not mask[1, 5]
    print("Challenge 6 passed!")
