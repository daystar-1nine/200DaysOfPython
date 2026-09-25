"""
Experiment sweeps and model comparison plots for Day 106.
Generates Charts 15 to 18.
"""

from pathlib import Path
from typing import Dict, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_experiment_and_comparison_charts(
    experiments_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    output_dir: Union[str, Path]
) -> None:
    """Generate Charts 15 to 18 and save as PNG images."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 15. Hidden units vs F1 (Filter Experiments A, B, C)
    h_df = experiments_df[experiments_df["experiment"].isin(["Exp_A_Compact_32", "Exp_B_Baseline_64", "Exp_C_HighCap_128"])].copy()
    plt.figure(figsize=(7, 5))
    sns.barplot(x="hidden_units", y="f1", data=h_df, palette="Blues_d")
    plt.title("Chart 15: SimpleRNN Hidden Units vs Test F1 Score", fontsize=14, fontweight="bold")
    plt.xlabel("Hidden State Dimension (H)")
    plt.ylabel("Test F1 Score")
    plt.ylim(0.85, 1.02)
    plt.tight_layout()
    plt.savefig(out / "15_hidden_units_vs_f1.png", dpi=150)
    plt.close()

    # 16. Sequence length vs F1 (Filter Experiment D)
    seq_df = experiments_df[experiments_df["experiment"].str.startswith("Exp_D_SeqLen_")].copy()
    plt.figure(figsize=(7, 5))
    plt.plot(seq_df["sequence_length"], seq_df["f1"], marker="o", color="#c0392b", lw=2.5)
    plt.title("Chart 16: Max Sequence Length vs Test F1 Score", fontsize=14, fontweight="bold")
    plt.xlabel("Sequence Length Cutoff (T)")
    plt.ylabel("Test F1 Score")
    plt.ylim(0.85, 1.02)
    plt.tight_layout()
    plt.savefig(out / "16_sequence_length_vs_f1.png", dpi=150)
    plt.close()

    # 17. Embedding dimension vs F1 (Experiments A, B, C)
    plt.figure(figsize=(7, 5))
    plt.plot(h_df["embedding_dim"], h_df["f1"], marker="s", color="#16a085", lw=2.5)
    plt.title("Chart 17: Embedding Dimension vs Test F1 Score", fontsize=14, fontweight="bold")
    plt.xlabel("Word Embedding Dimension (D)")
    plt.ylabel("Test F1 Score")
    plt.ylim(0.85, 1.02)
    plt.tight_layout()
    plt.savefig(out / "17_embedding_dimension_vs_f1.png", dpi=150)
    plt.close()

    # 18. Model Comparison
    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(x="model", y="f1", data=benchmark_df, palette="viridis")
    for p in ax.patches:
        val = p.get_height()
        ax.annotate(f"{val:.4f}", (p.get_x() + p.get_width() / 2., val / 2),
                    ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    plt.title("Chart 18: Model Comparison Benchmark (Test F1 Score)", fontsize=14, fontweight="bold")
    plt.xlabel("Model Architecture")
    plt.ylabel("Test F1 Score")
    plt.ylim(0.0, 1.1)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out / "18_model_comparison.png", dpi=150)
    plt.close()
