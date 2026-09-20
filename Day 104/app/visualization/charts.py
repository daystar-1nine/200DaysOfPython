"""
Visualization module for Day 104 Semantic Search Engine.
Generates all 12 required analytical visualizations and plots.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


class SearchVisualizer:
    """Produces publication-quality plots for search engine evaluation and analysis."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

    def plot_category_distribution(self, documents: List[Dict[str, Any]], filename: str = "1_category_distribution.png") -> Path:
        """Chart 1: Document category distribution."""
        categories = [d.get("category", "Unknown") for d in documents]
        counts = pd.Series(categories).value_counts()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=counts.values, y=counts.index, hue=counts.index, ax=ax, palette="viridis", legend=False)
        ax.set_title("1. Document Distribution Across Categories", fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of Documents")
        ax.set_ylabel("Category")

        for i, v in enumerate(counts.values):
            ax.text(v + 0.2, i, str(v), color="black", va="center", fontweight="bold")

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_document_length_distribution(self, documents: List[Dict[str, Any]], filename: str = "2_document_length_distribution.png") -> Path:
        """Chart 2: Document length distribution (word count)."""
        lengths = [len(f"{d.get('title', '')} {d.get('text', '')}".split()) for d in documents]

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(lengths, kde=True, ax=ax, color="#2b5c8f", bins=15)
        ax.set_title("2. Document Length Distribution (Word Count)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Word Count")
        ax.set_ylabel("Document Count")
        ax.axvline(np.mean(lengths), color="red", linestyle="--", label=f"Mean: {np.mean(lengths):.1f} words")
        ax.legend()

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_query_length_distribution(self, queries: List[Dict[str, Any]], filename: str = "3_query_length_distribution.png") -> Path:
        """Chart 3: Query length distribution."""
        lengths = [len(q.get("query", "").split()) for q in queries]

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(lengths, discrete=True, ax=ax, color="#17a2b8")
        ax.set_title("3. Benchmark Query Length Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel("Query Word Count")
        ax.set_ylabel("Number of Queries")
        ax.axvline(np.mean(lengths), color="orange", linestyle="--", label=f"Mean: {np.mean(lengths):.1f} words")
        ax.legend()

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_similarity_score_distribution(
        self,
        tfidf_scores: np.ndarray,
        emb_scores: np.ndarray,
        filename: str = "4_similarity_score_distribution.png"
    ) -> Path:
        """Chart 4: Similarity score distribution comparison."""
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.kdeplot(tfidf_scores.flatten(), ax=ax, label="TF-IDF Cosine Scores", fill=True, alpha=0.3, color="#007bff")
        sns.kdeplot(emb_scores.flatten(), ax=ax, label="Embedding Cosine Scores", fill=True, alpha=0.3, color="#28a745")
        ax.set_title("4. Raw Similarity Score Distributions", fontsize=12, fontweight="bold")
        ax.set_xlabel("Cosine Similarity Score")
        ax.set_ylabel("Density")
        ax.legend()

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_precision_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        filename: str = "5_precision_comparison.png"
    ) -> Path:
        """Chart 5: Precision@K comparison across engines."""
        k_labels = ["Precision@1", "Precision@3", "Precision@5"]
        data = []
        for engine, metrics in metrics_dict.items():
            for k_lbl in k_labels:
                data.append({
                    "Engine": engine,
                    "Metric": k_lbl,
                    "Score": metrics.get(f"Mean {k_lbl}", 0.0)
                })

        df = pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=df, x="Metric", y="Score", hue="Engine", ax=ax, palette="Set2")
        ax.set_title("5. Precision@K Comparison Across Engines", fontsize=12, fontweight="bold")
        ax.set_ylabel("Precision Score")
        ax.set_ylim(0.0, 1.05)

        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h),
                            ha="center", va="bottom", fontsize=8, xytext=(0, 2),
                            textcoords="offset points")

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_recall_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        filename: str = "6_recall_comparison.png"
    ) -> Path:
        """Chart 6: Recall@5 comparison across engines."""
        engines = list(metrics_dict.keys())
        scores = [metrics_dict[e].get("Mean Recall@5", 0.0) for e in engines]

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = sns.barplot(x=engines, y=scores, hue=engines, ax=ax, palette="Blues_d", legend=False)
        ax.set_title("6. Mean Recall@5 Comparison", fontsize=12, fontweight="bold")
        ax.set_ylabel("Recall@5")
        ax.set_ylim(0.0, 1.05)

        for bar, score in zip(bars.patches, scores):
            ax.annotate(f"{score:.3f}", (bar.get_x() + bar.get_width() / 2., score),
                        ha="center", va="bottom", fontsize=9, xytext=(0, 3),
                        textcoords="offset points")

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_mrr_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        filename: str = "7_mrr_comparison.png"
    ) -> Path:
        """Chart 7: Mean Reciprocal Rank (MRR) comparison."""
        engines = list(metrics_dict.keys())
        mrr_scores = [metrics_dict[e].get("MRR", 0.0) for e in engines]

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = sns.barplot(x=engines, y=mrr_scores, hue=engines, ax=ax, palette="Purples_d", legend=False)
        ax.set_title("7. Mean Reciprocal Rank (MRR) Comparison", fontsize=12, fontweight="bold")
        ax.set_ylabel("MRR")
        ax.set_ylim(0.0, 1.05)

        for bar, score in zip(bars.patches, mrr_scores):
            ax.annotate(f"{score:.3f}", (bar.get_x() + bar.get_width() / 2., score),
                        ha="center", va="bottom", fontsize=9, xytext=(0, 3),
                        textcoords="offset points")

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_tfidf_vs_embedding_correlation(
        self,
        tfidf_scores: np.ndarray,
        emb_scores: np.ndarray,
        filename: str = "8_tfidf_vs_embedding_correlation.png"
    ) -> Path:
        """Chart 8: TF-IDF vs Embedding score correlation scatter plot."""
        t_flat = tfidf_scores.flatten()
        e_flat = emb_scores.flatten()
        
        # Subsample if large
        if len(t_flat) > 2000:
            idx = np.random.choice(len(t_flat), size=2000, replace=False)
            t_flat, e_flat = t_flat[idx], e_flat[idx]

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(t_flat, e_flat, alpha=0.35, color="#6f42c1", edgecolors="none")
        corr = float(np.corrcoef(t_flat, e_flat)[0, 1]) if len(t_flat) > 1 else 0.0
        ax.set_title(f"8. TF-IDF vs Embedding Score Correlation (r = {corr:.2f})", fontsize=12, fontweight="bold")
        ax.set_xlabel("TF-IDF Cosine Score")
        ax.set_ylabel("Embedding Cosine Score")

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_top_retrieved_scores(
        self,
        top_scores_df: pd.DataFrame,
        filename: str = "9_top_retrieved_scores.png"
    ) -> Path:
        """Chart 9: Top retrieved documents score distribution."""
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.boxplot(data=top_scores_df, x="rank", y="score", hue="engine", ax=ax, palette="Set1")
        ax.set_title("9. Score Degradation by Rank Position", fontsize=12, fontweight="bold")
        ax.set_xlabel("Rank Position")
        ax.set_ylabel("Similarity Score")

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_error_analysis(
        self,
        error_df: pd.DataFrame,
        filename: str = "10_error_analysis.png"
    ) -> Path:
        """Chart 10: Retrieval error analysis by failure mode."""
        counts = error_df["failure_mode"].value_counts()

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(x=counts.values, y=counts.index, hue=counts.index, ax=ax, palette="Reds_r", legend=False)
        ax.set_title("10. Search Error Breakdown by Failure Mode", fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of Queries Affected")
        ax.set_ylabel("Failure Mode")

        for i, v in enumerate(counts.values):
            ax.text(v + 0.1, i, str(v), color="black", va="center", fontweight="bold")

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_embedding_pca(
        self,
        doc_matrix: np.ndarray,
        documents: List[Dict[str, Any]],
        filename: str = "11_embedding_pca_2d.png"
    ) -> Path:
        """Chart 11: 2D PCA projection of document embeddings colored by category."""
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(doc_matrix)

        categories = [d.get("category", "Unknown") for d in documents]
        df_pca = pd.DataFrame({
            "x": reduced[:, 0],
            "y": reduced[:, 1],
            "category": categories
        })

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(
            data=df_pca, x="x", y="y", hue="category", style="category",
            s=90, alpha=0.85, ax=ax, palette="tab10"
        )
        var_ratio = pca.explained_variance_ratio_
        ax.set_title(
            f"11. Document Embedding PCA (Explained Var: {var_ratio[0]*100:.1f}% + {var_ratio[1]*100:.1f}%)",
            fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_query_doc_heatmap(
        self,
        heatmap_matrix: np.ndarray,
        query_labels: List[str],
        doc_labels: List[str],
        filename: str = "12_query_doc_similarity_heatmap.png"
    ) -> Path:
        """Chart 12: Query-document similarity heatmap."""
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            heatmap_matrix,
            xticklabels=doc_labels,
            yticklabels=query_labels,
            cmap="mako",
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Cosine Similarity"},
            ax=ax
        )
        ax.set_title("12. Query vs Document Similarity Heatmap", fontsize=13, fontweight="bold")
        ax.set_xlabel("Documents")
        ax.set_ylabel("Queries")
        plt.xticks(rotation=45, ha="right")

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path
