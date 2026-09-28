"""
Multi-head attention heatmap visualizer across Transformer layers and heads.
"""
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_single_head_attention(
    tokens: List[str],
    weights: np.ndarray,
    title: str = "Attention Head",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Renders an attention heatmap for a single head.
    tokens: list of token strings up to actual message length.
    weights: (seq_len, seq_len) slice matching tokens.
    """
    n = len(tokens)
    sub_weights = weights[:n, :n]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.55), max(5, n * 0.45)))
    sns.heatmap(
        sub_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="Blues",
        annot=(n <= 12),
        fmt=".2f",
        cbar=True,
        ax=ax
    )
    ax.set_xlabel("Key Tokens", fontweight="bold")
    ax.set_ylabel("Query Tokens", fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig


def plot_multi_head_attention_grid(
    tokens: List[str],
    layer_weights: np.ndarray,
    layer_idx: int = 1,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Renders a 2x2 or 1xH grid of all attention heads in a given Transformer layer.
    layer_weights: (num_heads, seq_len, seq_len)
    """
    num_heads = layer_weights.shape[0]
    n = len(tokens)
    ncols = min(4, num_heads)
    nrows = int(np.ceil(num_heads / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * max(4, n * 0.4), nrows * max(3.5, n * 0.35)))
    if num_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for h in range(num_heads):
        ax = axes[h]
        sub = layer_weights[h, :n, :n]
        sns.heatmap(
            sub,
            xticklabels=tokens if h >= (nrows - 1) * ncols else False,
            yticklabels=tokens if h % ncols == 0 else False,
            cmap="Blues",
            cbar=False,
            ax=ax
        )
        ax.set_title(f"Head {h+1}", fontweight="bold", fontsize=10)
        if h >= (nrows - 1) * ncols:
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        if h % ncols == 0:
            ax.set_yticklabels(tokens, rotation=0, fontsize=8)

    # Hide any unused subplots
    for h in range(num_heads, len(axes)):
        axes[h].axis("off")

    fig.suptitle(f"Transformer Layer {layer_idx} — Multi-Head Self-Attention", fontweight="bold", fontsize=13)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig
