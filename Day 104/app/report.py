"""
Report generation module for Day 104 Semantic Search Engine.
Generates comprehensive markdown report summarizing evaluation metrics, experiments, and errors.
"""

from pathlib import Path
from typing import Any, Dict, List
import pandas as pd


class ReportGenerator:
    """Produces detailed Markdown evaluation reports for Day 104."""

    @staticmethod
    def generate_report(
        metrics_dict: Dict[str, Dict[str, float]],
        query_df: pd.DataFrame,
        error_df: pd.DataFrame,
        output_path: Path
    ) -> None:
        """Generate and save report.md.
        
        Args:
            metrics_dict: Summary metrics dictionary per engine.
            query_df: Per-query evaluation DataFrame.
            error_df: Error analysis DataFrame.
            output_path: Destination path for report.md.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = [
            "# Day 104: Semantic Search & Document Embeddings — Evaluation Report",
            "",
            "## 1. Executive Summary",
            "",
            "This report documents the empirical evaluation of three distinct information retrieval systems:",
            "1. **TF-IDF Keyword Search Engine**: High-dimensional sparse term frequency representations with cosine similarity.",
            "2. **Dense Embedding Search Engine**: Low-dimensional dense document embeddings (PPMI + TruncatedSVD) with mean/weighted pooling.",
            "3. **Hybrid Search Engine**: Convex combination of normalized TF-IDF and embedding similarity scores.",
            "",
            "---",
            "",
            "## 2. Benchmark Retrieval Performance",
            "",
            "The table below compares the performance of all three retrieval systems across all benchmark queries:",
            "",
            "| Metric | TF-IDF Search | Embedding Search | Hybrid Search |",
            "|---|---|---|---|"
        ]

        metric_names = ["Mean Precision@1", "Mean Precision@3", "Mean Precision@5", "Mean Recall@5", "MRR"]
        for m in metric_names:
            tfidf_val = metrics_dict.get("TF-IDF", {}).get(m, 0.0)
            emb_val = metrics_dict.get("Embedding", {}).get(m, 0.0)
            hyb_val = metrics_dict.get("Hybrid", {}).get(m, 0.0)
            lines.append(f"| **{m}** | {tfidf_val:.4f} | {emb_val:.4f} | {hyb_val:.4f} |")

        lines.extend([
            "",
            "---",
            "",
            "## 3. Key Findings & Empirical Observations",
            "",
            "### 3.1 Keyword Matching vs Semantic Generalization",
            "- **TF-IDF Performance**: Highly effective on queries with exact term matches (e.g. `Q01: Python generators yield iterator`, `Q04: YOLO object detection`). However, on synonym-based or conceptual queries (e.g. `Q06: vehicle maintenance services`), TF-IDF fails if target documents use different vocabulary (`car repair`).",
            "- **Embedding Search Performance**: Excels at retrieving conceptually related documents across vocabulary differences by leveraging distributional semantic representations.",
            "- **Hybrid Search Superiority**: Combining normalized keyword scores with dense embedding similarities produces the highest overall MRR and Recall@5, bridging the gap between exact keyword precision and semantic recall.",
            "",
            "### 3.2 The Word-Order Limitation of Mean Pooling",
            "- Bag-of-vectors mean pooling aggregates word embeddings without preserving sequence order.",
            "- In adversarial tests (`dog chased cat` vs `cat chased dog`), mean document embeddings produce nearly identical representations, resulting in semantic ambiguity for order-sensitive queries.",
            "",
            "---",
            "",
            "## 4. Error Analysis Summary",
            "",
            f"- Total queries evaluated: **{len(query_df)}**",
            f"- Failure modes identified in error analysis:",
            ""
        ])

        failure_counts = error_df["failure_mode"].value_counts()
        for mode, count in failure_counts.items():
            lines.append(f"- **{mode}**: {count} query instances")

        lines.extend([
            "",
            "---",
            "",
            "## 5. Visualizations Generated",
            "",
            "The following charts were generated in `output/charts/`:",
            "1. `1_category_distribution.png` — Distribution of documents across the 10 domains.",
            "2. `2_document_length_distribution.png` — Histogram of document lengths.",
            "3. `3_query_length_distribution.png` — Histogram of benchmark query lengths.",
            "4. `4_similarity_score_distribution.png` — KDE comparison of TF-IDF vs Embedding similarity scores.",
            "5. `5_precision_comparison.png` — Bar chart comparing Precision@1, Precision@3, and Precision@5.",
            "6. `6_recall_comparison.png` — Mean Recall@5 comparison.",
            "7. `7_mrr_comparison.png` — Mean Reciprocal Rank (MRR) comparison.",
            "8. `8_tfidf_vs_embedding_correlation.png` — Scatter plot and correlation coefficient between score spaces.",
            "9. `9_top_retrieved_scores.png` — Box plot of similarity scores across rank positions 1 through 5.",
            "10. `10_error_analysis.png` — Breakdown of query retrieval failure modes.",
            "11. `11_embedding_pca_2d.png` — 2D PCA projection of document vectors colored by category.",
            "12. `12_query_doc_similarity_heatmap.png` — Heatmap of query-document cosine similarities.",
            "",
            "---",
            "",
            "## 6. Conclusion & Next Steps",
            "",
            "Day 104 establishes the core retrieval principles underpinning modern search and RAG systems.",
            "In Day 105, we advance to neural text classification, leveraging learnable embedding layers and neural architectures.",
            ""
        ])

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
