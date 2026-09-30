"""
Plotting and visualization routines for Day 113 Scaling Lab experiments.
Generates publication-quality charts for:
1. Model Size vs Validation Loss (Scaling Law)
2. Model Size vs Training Time
3. Throughput (Tokens/sec) across Scaling Tiers
4. Learning Rate Schedule Comparison (Constant vs Cosine Warmup)
5. Data Quality Impact (Clean vs 4x Duplicated)
6. Data Leakage Contamination Impact
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_all_experiments(metrics_dir: Path, plots_dir: Path) -> None:
    metrics_dir = Path(metrics_dir)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Model Size vs Validation Loss
    scaling_csv = metrics_dir / "scaling_results.csv"
    if scaling_csv.exists():
        df_scaling = pd.read_csv(scaling_csv)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_scaling["parameters"] / 1e6, df_scaling["val_loss"], marker="o", linewidth=2.5, markersize=8, color="#1f77b4", label="Observed Validation Loss")
        for _, row in df_scaling.iterrows():
            ax.annotate(
                f"{row['model_name']} ({row['val_loss']:.2f})",
                (row["parameters"] / 1e6, row["val_loss"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontweight="bold"
            )
        ax.set_title("Model Size vs Validation Loss (Day 113 Scaling)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Parameters (Millions)", fontsize=11)
        ax.set_ylabel("Validation Loss", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "model_size_vs_loss.png", dpi=300)
        plt.close()

        # 2. Model Size vs Training Time & FLOPs
        fig, ax1 = plt.subplots(figsize=(8, 5))
        color = "#2ca02c"
        ax1.set_xlabel("Parameters (Millions)", fontsize=11)
        ax1.set_ylabel("Training Time (seconds)", color=color, fontsize=11)
        bars = ax1.bar(df_scaling["parameters"] / 1e6, df_scaling["total_time_seconds"], width=1.0, color=color, alpha=0.7, label="Wall-clock Time (s)")
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color_flops = "#d62728"
        ax2.set_ylabel("Training Compute (FLOPs)", color=color_flops, fontsize=11)
        ax2.plot(df_scaling["parameters"] / 1e6, df_scaling["training_flops"], color=color_flops, marker="s", linewidth=2, label="FLOPs (6ND)")
        ax2.tick_params(axis="y", labelcolor=color_flops)

        plt.title("Model Size vs Training Time & Compute", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(plots_dir / "model_size_vs_time.png", dpi=300)
        plt.close()

        # 3. Throughput Comparison (Tokens/sec)
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(df_scaling["model_name"], df_scaling["tokens_per_sec"], color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.55)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f} TPS",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center", va="bottom", fontweight="bold")
        ax.set_title("Training Throughput (Tokens / Second)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Tokens / Second", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6, axis="y")
        plt.tight_layout()
        plt.savefig(plots_dir / "throughput_comparison.png", dpi=300)
        plt.close()

    # 4. LR Schedule Comparison
    const_csv = metrics_dir / "sched_constant.csv"
    cosine_csv = metrics_dir / "sched_cosine.csv"
    if const_csv.exists() and cosine_csv.exists():
        df_const = pd.read_csv(const_csv)
        df_cosine = pd.read_csv(cosine_csv)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

        ax1.plot(df_const["step"], df_const["loss"], label="Constant LR Loss", color="#d62728", alpha=0.85)
        ax1.plot(df_cosine["step"], df_cosine["loss"], label="Warmup + Cosine Loss", color="#1f77b4", linewidth=2.0)
        ax1.set_title("Loss Convergence: Constant LR vs Warmup + Cosine Decay", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Train Loss", fontsize=11)
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.6)

        ax2.plot(df_const["step"], df_const["learning_rate"], label="Constant LR", color="#d62728", linestyle="--")
        ax2.plot(df_cosine["step"], df_cosine["learning_rate"], label="Cosine Schedule", color="#1f77b4", linewidth=2.0)
        ax2.set_title("Learning Rate Trajectory", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Training Steps", fontsize=11)
        ax2.set_ylabel("Learning Rate", fontsize=11)
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(plots_dir / "lr_schedule_comparison.png", dpi=300)
        plt.close()

    # 5. Data Quality Impact
    dupe_csv = metrics_dir / "data_quality_results.csv"
    if dupe_csv.exists():
        df_dupe = pd.read_csv(dupe_csv)
        fig, ax = plt.subplots(figsize=(7, 5))
        x_pos = np.arange(len(df_dupe))
        width = 0.35
        ax.bar(x_pos - width/2, df_dupe["final_train_loss"], width, label="Train Loss", color="#2ca02c", alpha=0.8)
        ax.bar(x_pos + width/2, df_dupe["final_val_loss"], width, label="Validation Loss", color="#d62728", alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_dupe["dataset_condition"], fontsize=10)
        ax.set_title("Data Quality Impact: Clean vs Duplicated Dataset", fontsize=12, fontweight="bold")
        ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6, axis="y")
        plt.tight_layout()
        plt.savefig(plots_dir / "data_quality_impact.png", dpi=300)
        plt.close()

    # 6. Data Leakage Impact
    leak_csv = metrics_dir / "data_leakage_results.csv"
    if leak_csv.exists():
        df_leak = pd.read_csv(leak_csv)
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(df_leak["condition"], df_leak["apparent_val_loss"], color=["#1f77b4", "#e377c2"], width=0.5)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.4f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center", va="bottom", fontweight="bold")
        ax.set_title("Data Leakage Contamination: Apparent Validation Loss", fontsize=12, fontweight="bold")
        ax.set_ylabel("Apparent Validation Loss (Lower = Illusion of Success)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6, axis="y")
        plt.tight_layout()
        plt.savefig(plots_dir / "leakage_impact.png", dpi=300)
        plt.close()
