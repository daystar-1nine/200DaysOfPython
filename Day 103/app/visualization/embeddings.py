"""Generates all 6 required Word2Vec analytical visualizations."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from app.data.vocabulary import Vocabulary

def generate_visualizations(
    losses: list,
    vocab: Vocabulary,
    W: np.ndarray,
    sim_matrix_df: pd.DataFrame,
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Training Loss (Epoch -> Loss)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(losses) + 1), losses, color="#2980b9", lw=2, marker="o", markersize=3)
    plt.title("1. Word2Vec Training Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Negative Sampling Loss")
    plt.savefig(output_dir / "1_training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Vocabulary Frequency
    top_words = pd.Series(vocab.get_frequency_distribution()).sort_values(ascending=False).head(15)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_words.values, y=top_words.index, hue=top_words.index, palette="Blues_r", legend=False)
    plt.title("2. Top 15 Vocabulary Terms by Frequency")
    plt.xlabel("Count")
    plt.savefig(output_dir / "2_vocabulary_frequency.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Word Frequency Distribution
    all_counts = list(vocab.word_counts.values())
    plt.figure(figsize=(8, 4))
    sns.histplot(all_counts, bins=15, kde=True, color="#8e44ad")
    plt.title("3. Word Frequency Distribution")
    plt.xlabel("Frequency Count")
    plt.savefig(output_dir / "3_word_frequency_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Embedding Norms
    norms = np.linalg.norm(W, axis=1)
    plt.figure(figsize=(8, 4))
    sns.histplot(norms, bins=15, kde=True, color="#27ae60")
    plt.title("4. Distribution of Learned Embedding Vector Norms")
    plt.xlabel("L2 Norm ||v||")
    plt.savefig(output_dir / "4_embedding_norms.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Similarity Heatmap
    if not sim_matrix_df.empty:
        plt.figure(figsize=(8, 7))
        sns.heatmap(sim_matrix_df, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1.0, vmax=1.0)
        plt.title("5. Pairwise Cosine Similarity Heatmap")
        plt.savefig(output_dir / "5_similarity_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 6. 2D Embedding Visualization via PCA
    if W.shape[0] >= 3:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(W)
        plt.figure(figsize=(10, 8))
        plt.scatter(coords[:, 0], coords[:, 1], color="#e74c3c", s=60, alpha=0.8)
        for i, word in enumerate(vocab.word2idx.keys()):
            plt.annotate(
                word,
                xy=(coords[i, 0], coords[i, 1]),
                xytext=(5, 2),
                textcoords="offset points",
                fontsize=9,
                weight="bold"
            )
        plt.title("6. 2D PCA Projection of Learned Word Embeddings")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        plt.savefig(output_dir / "6_embedding_pca_2d.png", dpi=150, bbox_inches="tight")
        plt.close()

    print("Generated all 6 Word2Vec visualizations successfully.")
