"""
Dimensionality reduction and visualization for BERT [CLS] contextual representations.
"""
from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def visualize_cls_embeddings(
    cls_embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: Optional[str] = None
) -> None:
    """
    Projects high-dimensional [CLS] representations onto 2D space using PCA
    and plots class separation between Ham and Spam.
    """
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(cls_embeddings)
    var_exp = pca.explained_variance_ratio_

    labels_arr = np.asarray(labels, dtype=int)
    ham_mask = labels_arr == 0
    spam_mask = labels_arr == 1

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        embeddings_2d[ham_mask, 0],
        embeddings_2d[ham_mask, 1],
        c="#3498db",
        label=f"Ham (N={ham_mask.sum()})",
        alpha=0.6,
        edgecolors="none",
        s=35
    )
    ax.scatter(
        embeddings_2d[spam_mask, 0],
        embeddings_2d[spam_mask, 1],
        c="#e74c3c",
        label=f"Spam (N={spam_mask.sum()})",
        alpha=0.8,
        edgecolors="k",
        linewidths=0.5,
        s=45
    )

    resolved_title = title or "BERT [CLS] Embedding Space (PCA Projection)"
    ax.set_title(resolved_title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(f"Principal Component 1 ({var_exp[0]*100:.1f}% Variance)", fontsize=11)
    ax.set_ylabel(f"Principal Component 2 ({var_exp[1]*100:.1f}% Variance)", fontsize=11)
    ax.legend(frameon=True, facecolor="#f8f9fa", edgecolor="#ced4da", loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[CLS] Embeddings PCA plot saved to: {output_path}")
