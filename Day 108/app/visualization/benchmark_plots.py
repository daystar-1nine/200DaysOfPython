import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def generate_benchmark_plots(
    comparison_df: pd.DataFrame,
    units_df: pd.DataFrame,
    seqlen_df: pd.DataFrame,
    dropout_df: pd.DataFrame,
    charts_dir: Path
):
    """
    Generates benchmarking visualizations:
      Plot 16: Parameter count
      Plot 17: Training time
      Plot 18: F1 comparison
      Plot 19: Hidden units vs F1
      Plot 20: Sequence length vs F1
      Plot 21: Dropout vs F1
      Plot 22: Model Efficiency Plot (Parameters vs F1 Score)
    """
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 16: Parameter Count
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(
        comparison_df["model"], comparison_df["parameters"],
        color=sns.color_palette("muted", len(comparison_df)), edgecolor="black"
    )
    plt.title("Plot 16: Trainable Parameters by Recurrent Architecture", fontsize=12, fontweight="bold")
    plt.ylabel("Parameters")
    plt.xticks(rotation=20)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 500, f"{int(h):,}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(charts_dir / "16_parameter_count_comparison.png", dpi=200)
    plt.close()
    
    # Plot 17: Training Time
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(
        comparison_df["model"], comparison_df["training_time_sec"],
        color="#ff7f0e", edgecolor="black", alpha=0.85
    )
    plt.title("Plot 17: Training Time Comparison (Seconds)", fontsize=12, fontweight="bold")
    plt.ylabel("Time (Seconds)")
    plt.xticks(rotation=20)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.1, f"{h:.2f}s", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(charts_dir / "17_training_time_comparison.png", dpi=200)
    plt.close()
    
    # Plot 18: F1 Comparison
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(
        comparison_df["model"], comparison_df["f1"],
        color="#2ca02c", edgecolor="black", alpha=0.85
    )
    plt.title("Plot 18: F1 Score Comparison", fontsize=12, fontweight="bold")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.01, f"{h:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(charts_dir / "18_f1_score_comparison.png", dpi=200)
    plt.close()
    
    # Plot 19: Hidden Units vs F1
    if not units_df.empty:
        plt.figure(figsize=(7, 4.5))
        for model_name, grp in units_df.groupby("architecture"):
            plt.plot(grp["units"], grp["f1"], marker="o", lw=2, label=model_name)
        plt.title("Plot 19: Hidden Units vs F1 Score", fontsize=12, fontweight="bold")
        plt.xlabel("Hidden Units (32, 64, 128)")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(charts_dir / "19_hidden_units_vs_f1.png", dpi=200)
        plt.close()
        
    # Plot 20: Sequence Length vs F1
    if not seqlen_df.empty:
        plt.figure(figsize=(7, 4.5))
        plt.plot(seqlen_df["seq_len"], seqlen_df["f1"], marker="s", color="#d62728", lw=2)
        plt.title("Plot 20: Sequence Length Sweep vs F1 Score", fontsize=12, fontweight="bold")
        plt.xlabel("Sequence Length (20, 40, 60, 100)")
        plt.ylabel("F1 Score")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(charts_dir / "20_sequence_length_vs_f1.png", dpi=200)
        plt.close()
        
    # Plot 21: Dropout vs F1
    if not dropout_df.empty:
        plt.figure(figsize=(7, 4.5))
        plt.plot(dropout_df["dropout"], dropout_df["f1"], marker="^", color="#9467bd", lw=2)
        plt.title("Plot 21: GRU Dropout Sweep vs F1 Score", fontsize=12, fontweight="bold")
        plt.xlabel("Dropout Rate (0.0, 0.2, 0.3, 0.5)")
        plt.ylabel("F1 Score")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(charts_dir / "21_dropout_vs_f1.png", dpi=200)
        plt.close()
        
    # Plot 22: Model Efficiency Plot (Section 31: X = Parameters, Y = F1 Score)
    plt.figure(figsize=(8, 5.5))
    plt.scatter(
        comparison_df["parameters"], comparison_df["f1"],
        s=120, color="#1f77b4", edgecolor="black", zorder=3
    )
    for _, row in comparison_df.iterrows():
        plt.annotate(
            row["model"],
            (row["parameters"], row["f1"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
            fontweight="bold"
        )
    plt.title("Plot 22: Model Efficiency Plot (F1 Score vs Parameter Complexity)", fontsize=12, fontweight="bold")
    plt.xlabel("Trainable Parameter Count")
    plt.ylabel("Test F1 Score")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(charts_dir / "22_model_efficiency_scatter.png", dpi=200)
    plt.close()
