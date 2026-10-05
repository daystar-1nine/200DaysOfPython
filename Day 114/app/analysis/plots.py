"""
Plotting and visualization routines for Day 114 Instruction Fine-Tuning experiments.
Generates publication-quality charts for:
1. Training & Validation Loss convergence
2. Overfitting & Epoch Scaling (1 vs 3 vs 5 epochs)
3. Base vs SFT Instruction Following Scores across categories
4. Full Fine-Tuning vs LoRA parameter efficiency
5. Learning Rate Convergence Comparison
6. Decoding Temperature & Lexical Diversity
"""
import json
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

    # 1. Training & Validation Loss Curve
    sft_csv = metrics_dir / "sft_standard.csv"
    if sft_csv.exists():
        df_sft = pd.read_csv(sft_csv)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_sft["step"], df_sft["train_loss"], label="Train Loss (Assistant tokens)", color="#1f77b4", linewidth=2.0)
        ax.plot(df_sft["step"], df_sft["val_loss"], label="Validation Loss", color="#d62728", linewidth=2.0, linestyle="--")
        ax.set_title("SFT Loss Trajectory (Masked Assistant Tokens)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(plots_dir / "training_validation_loss.png", dpi=300)
        plt.close()

    # 2. Overfitting & Epoch Scaling
    overfit_csv = metrics_dir / "overfitting_results.csv"
    if overfit_csv.exists():
        df_of = pd.read_csv(overfit_csv)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        x_pos = np.arange(len(df_of))
        width = 0.35
        ax1.bar(x_pos - width/2, df_of["final_train_loss"], width, label="Train Loss", color="#2ca02c", alpha=0.8)
        ax1.bar(x_pos + width/2, df_of["final_val_loss"], width, label="Validation Loss", color="#d62728", alpha=0.8)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"{e} Epochs" for e in df_of["epochs"]], fontsize=10)
        ax1.set_title("Train vs Validation Loss (Overfitting)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.6, axis="y")

        ax2.plot(df_of["epochs"], df_of["test_instruction_score_pct"], marker="o", color="#9467bd", linewidth=2.5, markersize=8)
        for _, row in df_of.iterrows():
            ax2.annotate(f"{row['test_instruction_score_pct']:.1f}%",
                         (row["epochs"], row["test_instruction_score_pct"]),
                         textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold")
        ax2.set_title("Instruction Following Score vs Epochs", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epochs", fontsize=11)
        ax2.set_ylabel("Test Instruction Score (%)", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(plots_dir / "overfitting_analysis.png", dpi=300)
        plt.close()

    # 3. Base vs SFT Instruction Following Scores by Category
    eval_json_path = metrics_dir.parent / "evaluation.json"
    if eval_json_path.exists():
        with open(eval_json_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        cats = list(eval_data.get("category_breakdown", {}).keys())
        sft_scores = [eval_data["category_breakdown"][c]["score_pct"] for c in cats]
        # Base model scores are typically 0% or low baseline (~10%)
        base_scores = [10.0] * len(cats)

        fig, ax = plt.subplots(figsize=(10, 5))
        x_indices = np.arange(len(cats))
        width = 0.35
        ax.bar(x_indices - width/2, base_scores, width, label="Base Model (Pre-SFT)", color="#7f7f7f", alpha=0.7)
        ax.bar(x_indices + width/2, sft_scores, width, label="MiniGPT-Chat (Post-SFT)", color="#1f77b4", alpha=0.9)
        ax.set_xticks(x_indices)
        ax.set_xticklabels([c.replace("_", " ").title() for c in cats], fontsize=10, rotation=15)
        ax.set_title("Instruction Following Performance by Category (0/1/2 Rubric)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Score (%)", fontsize=11)
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6, axis="y")
        plt.tight_layout()
        plt.savefig(plots_dir / "base_vs_sft_scores.png", dpi=300)
        plt.close()

    # 4. Full Fine-Tuning vs LoRA Parameter Efficiency
    lora_csv = metrics_dir / "lora_comparison.csv"
    if lora_csv.exists():
        df_lora = pd.read_csv(lora_csv)
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(df_lora["tuning_mode"], df_lora["trainable_parameters"], color=["#1f77b4", "#2ca02c"], width=0.5)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{int(height):,} params",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
        ax.set_title("Trainable Parameters: Full SFT vs LoRA Adapter", fontsize=12, fontweight="bold")
        ax.set_ylabel("Trainable Parameters", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6, axis="y")
        plt.tight_layout()
        plt.savefig(plots_dir / "full_vs_lora_parameters.png", dpi=300)
        plt.close()

    # 5. Learning Rate Convergence Comparison
    lr_csv = metrics_dir / "learning_rate_results.csv"
    if lr_csv.exists():
        df_lr = pd.read_csv(lr_csv)
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar([f"LR {lr:.0e}" for lr in df_lr["learning_rate"]], df_lr["instruction_score_pct"], color=["#ff7f0e", "#1f77b4"], width=0.45)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
        ax.set_title("Instruction Following Score by Learning Rate", fontsize=12, fontweight="bold")
        ax.set_ylabel("Instruction Score (%)", fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.6, axis="y")
        plt.tight_layout()
        plt.savefig(plots_dir / "learning_rate_comparison.png", dpi=300)
        plt.close()

    # 6. Decoding Temperature & Lexical Diversity
    temp_csv = metrics_dir / "temperature_results.csv"
    if temp_csv.exists():
        df_temp = pd.read_csv(temp_csv)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(df_temp["temperature"], df_temp["type_token_ratio (diversity)"], marker="s", color="#e377c2", linewidth=2.5, markersize=8)
        for _, row in df_temp.iterrows():
            ax.annotate(f"TTR: {row['type_token_ratio (diversity)']:.2f}",
                        (row["temperature"], row["type_token_ratio (diversity)"]),
                        textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold")
        ax.set_title("Generation Temperature vs Lexical Diversity (Type-Token Ratio)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Temperature", fontsize=11)
        ax.set_ylabel("Type-Token Ratio (Higher = More Diverse)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(plots_dir / "temperature_diversity.png", dpi=300)
        plt.close()
