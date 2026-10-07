"""
Publication-quality chart generator for Day 116 LLM Evaluation Harness.
Produces 12 high-resolution, beautifully styled visualization artifacts.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set consistent, modern publication styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "figure.dpi": 300
})

COLORS = {
    "BASE": "#64748B",  # Slate gray
    "SFT": "#3B82F6",   # Vibrant blue
    "DPO": "#10B981",   # Emerald green
    "danger": "#EF4444",
    "warning": "#F59E0B",
    "info": "#6366F1"
}


def chart_overall_comparison(df_comp: pd.DataFrame, out_path: Path) -> None:
    """1. Overall composite benchmark score."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df_comp["model"], df_comp["overall_score"], color=[COLORS[m] for m in df_comp["model"]], width=0.55, edgecolor="black", linewidth=1.2)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_title("Overall Model Evaluation Score (Composite Metric)", pad=15)
    ax.set_ylabel("Composite Score [0 - 100]")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def chart_instruction_following(df_comp: pd.DataFrame, out_path: Path) -> None:
    """2. Instruction following adherence score."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df_comp["model"], df_comp["instruction_score"], color=[COLORS[m] for m in df_comp["model"]], width=0.55, edgecolor="black", linewidth=1.2)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_title("Instruction Following & Constraint Compliance Rate", pad=15)
    ax.set_ylabel("Compliance Score (%)")
    ax.set_ylim(80, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def chart_factuality_hallucination(df_comp: pd.DataFrame, out_path: Path) -> None:
    """3. Factuality and hallucination abstention score."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df_comp["model"], df_comp["factuality"], color=[COLORS[m] for m in df_comp["model"]], width=0.55, edgecolor="black", linewidth=1.2)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_title("Factuality Score & Abstention on Unverifiable Claims", pad=15)
    ax.set_ylabel("Factuality Score (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def chart_coding_pass_rate(df_comp: pd.DataFrame, out_path: Path) -> None:
    """4. Coding benchmark pass rate."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df_comp["model"], df_comp["coding_pass_rate"], color=[COLORS[m] for m in df_comp["model"]], width=0.55, edgecolor="black", linewidth=1.2)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_title("Sandboxed Coding Test Pass Rate (AST & Unit Tests)", pad=15)
    ax.set_ylabel("Pass Rate (%)")
    ax.set_ylim(0, 50)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def chart_safety_score(df_comp: pd.DataFrame, out_path: Path) -> None:
    """5. Safety compliance and refusal score."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df_comp["model"], df_comp["safety"], color=[COLORS[m] for m in df_comp["model"]], width=0.55, edgecolor="black", linewidth=1.2)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_title("Safety Policy Guardrail Compliance & Safe Refusal Rate", pad=15)
    ax.set_ylabel("Safety Compliance (%)")
    ax.set_ylim(70, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def chart_bleu_comparison(df_comp: pd.DataFrame, out_path: Path) -> None:
    """6. BLEU comparison with error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))
    models = df_comp["model"].tolist()
    bleu_vals = df_comp["bleu"].tolist()
    
    # Parse 95% CIs
    ci_lows = []
    ci_highs = []
    for ci_str in df_comp["bleu_ci_95"]:
        clean = str(ci_str).replace("[", "").replace("]", "").replace(" ", "")
        low, high = map(float, clean.split(","))
        ci_lows.append(low)
        ci_highs.append(high)

    yerr = [
        [b - l for b, l in zip(bleu_vals, ci_lows)],
        [h - b for h, b in zip(ci_highs, bleu_vals)]
    ]

    bars = ax.bar(models, bleu_vals, yerr=yerr, capsize=6, color=[COLORS[m] for m in models], width=0.55, edgecolor="black", linewidth=1.2)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_title("Corpus BLEU Comparison with 95% Bootstrap CI", pad=15)
    ax.set_ylabel("BLEU Score (%)")
    ax.set_ylim(0, 25)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def chart_rouge_comparison(df_comp: pd.DataFrame, out_path: Path) -> None:
    """7. Grouped ROUGE metric comparison (ROUGE-1, ROUGE-2, ROUGE-L)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(df_comp))
    width = 0.25

    rects1 = ax.bar(x - width, df_comp["rouge1"], width, label="ROUGE-1", color="#60A5FA", edgecolor="black")
    rects2 = ax.bar(x, df_comp["rouge2"], width, label="ROUGE-2", color="#F59E0B", edgecolor="black")
    rects3 = ax.bar(x + width, df_comp["rougeL"], width, label="ROUGE-L", color="#10B981", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(df_comp["model"])
    ax.set_ylabel("Score (%)")
    ax.set_title("ROUGE N-Gram Recall & Longest Common Subsequence", pad=15)
    ax.legend(frameon=True)
    ax.set_ylim(0, 70)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def chart_semantic_similarity(df_comp: pd.DataFrame, out_path: Path) -> None:
    """8. Semantic similarity comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df_comp["model"], df_comp["semantic_similarity"], color=[COLORS[m] for m in df_comp["model"]], width=0.55, edgecolor="black", linewidth=1.2)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_title("Semantic Cosine Similarity to Gold References", pad=15)
    ax.set_ylabel("Similarity (%)")
    ax.set_ylim(0, 70)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def chart_category_heatmap(predictions_path: Path, out_path: Path) -> None:
    """9. Category performance heatmap."""
    records = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    df = pd.DataFrame(records)
    # Composite per row: average of instruction_score, factuality, semantic_similarity
    df["composite"] = (df["instruction_score"] * 0.4 + df["factuality"] * 0.3 + df["semantic_similarity"] * 0.3).round(1)
    pivot = df.pivot_table(index="category", columns="model", values="composite", aggfunc="mean").round(1)
    pivot.columns = [c.upper() for c in pivot.columns]
    # Ensure standard order
    col_order = [m for m in ["BASE", "SFT", "DPO"] if m in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Score [0-100]'}, ax=ax, linewidths=1.0)
    ax.set_title("Performance Heatmap Across Benchmark Categories", pad=15)
    ax.set_xlabel("Model Architecture")
    ax.set_ylabel("Evaluation Category")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def chart_win_rate_donut(wr_df: pd.DataFrame, pair_prefix: str, title: str, out_path: Path) -> None:
    """10 & 11. Pairwise win rate donut chart."""
    match = wr_df[wr_df["model_a"] + "_VS_" + wr_df["model_b"] == pair_prefix.upper()]
    if match.empty:
        # Fallback search
        match = wr_df.iloc[0:1]
    
    row = match.iloc[0]
    labels = [f"{row['model_a']} Wins", f"{row['model_b']} Wins", "Ties"]
    sizes = [row["wins_a"], row["wins_b"], row["ties"]]
    colors = [COLORS.get(row['model_a'], "#64748B"), COLORS.get(row['model_b'], "#10B981"), "#CBD5E1"]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", startangle=140,
        colors=colors, pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_weight("bold")

    ax.set_title(title, pad=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def chart_error_distribution(error_df: pd.DataFrame, out_path: Path) -> None:
    """12. Failure category breakdown across models."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    counts = error_df.groupby(["failure_type", "model"]).size().unstack(fill_value=0)
    
    counts.plot(kind="bar", stacked=False, ax=ax, color=[COLORS.get(m.upper(), "#94A3B8") for m in counts.columns], edgecolor="black", width=0.8)
    ax.set_title("Error Taxonomy Distribution by Model", pad=15)
    ax.set_xlabel("Failure Mode Taxonomy")
    ax.set_ylabel("Count of Failure Instances")
    ax.legend(title="Model", frameon=True)
    plt.xticks(rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def generate_all_charts(
    metrics_dir: Path,
    raw_dir: Path,
    reports_dir: Path,
    charts_dir: Path
) -> List[Path]:
    """Generates and persists all 12 publication charts."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    df_comp = pd.read_csv(metrics_dir / "model_comparison.csv")
    df_wr = pd.read_csv(metrics_dir / "win_rates.csv")
    df_err = pd.read_csv(reports_dir / "error_analysis.csv")
    pred_path = raw_dir / "predictions.jsonl"

    generated = []

    chart_tasks = [
        ("overall_model_comparison.png", lambda p: chart_overall_comparison(df_comp, p)),
        ("instruction_following_score.png", lambda p: chart_instruction_following(df_comp, p)),
        ("factuality_and_hallucination.png", lambda p: chart_factuality_hallucination(df_comp, p)),
        ("coding_pass_rate.png", lambda p: chart_coding_pass_rate(df_comp, p)),
        ("safety_score_comparison.png", lambda p: chart_safety_score(df_comp, p)),
        ("bleu_comparison.png", lambda p: chart_bleu_comparison(df_comp, p)),
        ("rouge_comparison.png", lambda p: chart_rouge_comparison(df_comp, p)),
        ("semantic_similarity_comparison.png", lambda p: chart_semantic_similarity(df_comp, p)),
        ("category_performance_heatmap.png", lambda p: chart_category_heatmap(pred_path, p)),
        ("base_vs_sft_win_rate.png", lambda p: chart_win_rate_donut(df_wr, "BASE_VS_SFT", "Pairwise Win Rate: BASE vs SFT", p)),
        ("base_vs_dpo_win_rate.png", lambda p: chart_win_rate_donut(df_wr, "BASE_VS_DPO", "Pairwise Win Rate: BASE vs DPO", p)),
        ("error_category_distribution.png", lambda p: chart_error_distribution(df_err, p)),
    ]

    for fname, func in chart_tasks:
        target = charts_dir / fname
        func(target)
        generated.append(target)
        print(f"  [Chart Generated] {target.name}")

    return generated
