"""
Publication-Quality Visualization Suite for Advanced RAG Benchmarks.
Generates 15 comprehensive figures covering IR metrics, latency, ablation, diversity, and error distributions.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def setup_matplotlib():
    """Sets publication style settings for clean visualization."""
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#e0e0e0"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7


def chart_1_bm25_vs_dense_vs_hybrid(data: Dict[str, Dict[str, float]], output_path: Path):
    """Chart 1: Precision@5, Recall@5, MRR across BM25, Dense, and Hybrid."""
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    models = ["BM25", "Dense", "Hybrid (RRF)"]
    keys = ["bm25", "dense", "hybrid_rrf"]
    metrics = ["mean_p@5", "mean_r@5", "mean_mrr", "mean_ndcg@5"]
    metric_labels = ["Precision@5", "Recall@5", "MRR", "nDCG@5"]
    
    x = np.arange(len(models))
    width = 0.2
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]
    
    for i, (m_key, m_label, col) in enumerate(zip(metrics, metric_labels, colors)):
        vals = [data.get(k, {}).get(m_key, 0.0) for k in keys]
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, vals, width, label=m_label, color=col, alpha=0.9, edgecolor="#222222", linewidth=0.6)
        for r in rects:
            h = r.get_height()
            if h > 0:
                ax.annotate(f"{h:.2f}",
                            xy=(r.get_x() + r.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7, fontweight="bold")
                
    ax.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
    ax.set_title("Figure 1: BM25 vs Dense vs Hybrid (RRF) Retrieval Quality", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y")
    ax.legend(frameon=True, facecolor="white", loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def chart_2_weighted_vs_rrf(data: Dict[str, Dict[str, float]], output_path: Path):
    """Chart 2: Score Fusion vs RRF across K = 1, 3, 5, 10."""
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    k_vals = [1, 3, 5, 10]
    weighted_p = [data.get("hybrid_weighted", {}).get(f"mean_p@{k}", 0.0) for k in k_vals]
    rrf_p = [data.get("hybrid_rrf", {}).get(f"mean_p@{k}", 0.0) for k in k_vals]
    weighted_r = [data.get("hybrid_weighted", {}).get(f"mean_r@{k}", 0.0) for k in k_vals]
    rrf_r = [data.get("hybrid_rrf", {}).get(f"mean_r@{k}", 0.0) for k in k_vals]
    
    ax.plot(k_vals, weighted_p, marker="o", linewidth=2, color="#2980b9", linestyle="--", label="Weighted Score P@K")
    ax.plot(k_vals, rrf_p, marker="s", linewidth=2, color="#16a085", linestyle="-", label="RRF Fusion P@K")
    ax.plot(k_vals, weighted_r, marker="^", linewidth=2, color="#d35400", linestyle="--", label="Weighted Score Recall@K")
    ax.plot(k_vals, rrf_r, marker="D", linewidth=2, color="#c0392b", linestyle="-", label="RRF Fusion Recall@K")
    
    ax.set_xlabel("Top-K Cutoff", fontsize=11, fontweight="bold")
    ax.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
    ax.set_title("Figure 2: Weighted Score Fusion vs Reciprocal Rank Fusion (RRF)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(k_vals)
    ax.set_ylim(0, 1.1)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def chart_3_query_optimization_ablation(data: Dict[str, Dict[str, float]], output_path: Path):
    """Chart 3: Query Optimization ablation (Raw vs Rewritten vs Expanded vs Multi-query)."""
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    variants = ["Raw Query", "Rewritten", "Expanded", "Multi-Query RRF"]
    keys = ["raw", "rewritten", "expanded", "multi_query"]
    mrrs = [data.get(k, {}).get("mean_mrr", 0.0) for k in keys]
    recalls = [data.get(k, {}).get("mean_r@5", 0.0) for k in keys]
    
    x = np.arange(len(variants))
    w = 0.35
    
    r1 = ax.bar(x - w/2, mrrs, w, label="MRR", color="#34495e", edgecolor="#222222")
    r2 = ax.bar(x + w/2, recalls, w, label="Recall@5", color="#e67e22", edgecolor="#222222")
    
    for r in list(r1) + list(r2):
        h = r.get_height()
        ax.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
                    
    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("Figure 3: Impact of Query Expansion, Rewriting, and Multi-Query Search", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y")
    ax.legend(frameon=True, facecolor="white", loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def chart_4_reranking_ndcg_gain(data: Dict[str, Dict[str, float]], output_path: Path):
    """Chart 4: Cross-Encoder Reranking Gain on nDCG@5 and MRR."""
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    
    stages = ["Initial Hybrid Retriever", "After Cross-Encoder Reranking"]
    ndcg5 = [data.get("hybrid_rrf", {}).get("mean_ndcg@5", 0.0), data.get("hybrid_rrf_rerank", {}).get("mean_ndcg@5", 0.0)]
    mrr = [data.get("hybrid_rrf", {}).get("mean_mrr", 0.0), data.get("hybrid_rrf_rerank", {}).get("mean_mrr", 0.0)]
    
    x = np.arange(len(stages))
    w = 0.35
    
    r1 = ax.bar(x - w/2, ndcg5, w, label="nDCG@5", color="#8e44ad", edgecolor="#222222")
    r2 = ax.bar(x + w/2, mrr, w, label="MRR", color="#27ae60", edgecolor="#222222")
    
    for r in list(r1) + list(r2):
        h = r.get_height()
        ax.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
                    
    gain_ndcg = ((ndcg5[1] - ndcg5[0]) / max(1e-4, ndcg5[0])) * 100.0
    ax.text(0.5, 0.15, f"Cross-Encoder Gain: +{gain_ndcg:.1f}% nDCG@5",
            transform=ax.transAxes, ha="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", edgecolor="#bdc3c7"))
            
    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("Figure 4: Cross-Encoder Reranking Ranking Quality Lift", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y")
    ax.legend(frameon=True, facecolor="white", loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def chart_5_mmr_diversity_frontier(lambda_results: List[Dict[str, float]], output_path: Path):
    """Chart 5: MMR Diversity Trade-off Frontier (Relevance vs Novelty as lambda varies)."""
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    lambdas = [d["lambda"] for d in lambda_results]
    relevance = [d["relevance_mrr"] for d in lambda_results]
    diversity = [d["diversity_score"] for d in lambda_results]
    
    ax.plot(lambdas, relevance, marker="o", color="#2980b9", linewidth=2.5, label="Relevance (MRR)")
    ax.plot(lambdas, diversity, marker="s", color="#e74c3c", linewidth=2.5, label="Diversity (1 - Max Inter-Chunk Sim)")
    
    # Highlight optimal lambda = 0.7
    ax.axvline(x=0.7, color="#27ae60", linestyle=":", linewidth=1.5, label="Balanced Operating Point (lambda=0.7)")
    
    ax.set_xlabel(r"MMR Trade-off Parameter $\lambda$ (Relevance vs Diversity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Normalized Metric Score", fontsize=11, fontweight="bold")
    ax.set_title(r"Figure 5: Maximal Marginal Relevance (MMR) Pareto Diversity Frontier", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.0, 1.1)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", loc="center left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def chart_6_latency_vs_mrr_tradeoff(latencies: Dict[str, float], mrrs: Dict[str, float], output_path: Path):
    """Chart 6: Latency vs MRR Trade-off across Pipelines."""
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    models = list(latencies.keys())
    x_lat = [latencies[m] for m in models]
    y_mrr = [mrrs[m] for m in models]
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c", "#1abc9c"]
    
    for i, m in enumerate(models):
        ax.scatter(x_lat[i], y_mrr[i], s=160, color=colors[i % len(colors)], edgecolors="#222222", zorder=3)
        ax.annotate(m, (x_lat[i], y_mrr[i]), xytext=(7, -2), textcoords="offset points", fontsize=8, fontweight="bold")
        
    ax.set_xlabel("Median Latency per Query (ms)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Reciprocal Rank (MRR)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 6: Query Latency vs Retrieval Quality Trade-off", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def chart_7_context_compression_efficiency(compression_stats: Dict[str, Any], output_path: Path):
    """Chart 7: Original vs Compressed Context Tokens & Compression Ratio."""
    setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
    
    cats = list(compression_stats["by_category"].keys())
    orig_chars = [compression_stats["by_category"][c]["original"] for c in cats]
    comp_chars = [compression_stats["by_category"][c]["compressed"] for c in cats]
    ratios = [compression_stats["by_category"][c]["ratio"] * 100 for c in cats]
    
    x = np.arange(len(cats))
    w = 0.35
    
    ax1.bar(x - w/2, orig_chars, w, label="Original Characters", color="#95a5a6", edgecolor="#222222")
    ax1.bar(x + w/2, comp_chars, w, label="Compressed Characters", color="#2980b9", edgecolor="#222222")
    ax1.set_ylabel("Average Characters in Prompt", fontsize=10, fontweight="bold")
    ax1.set_title("Prompt Length Reduction", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, rotation=25, ha="right", fontsize=8)
    ax1.grid(True, axis="y")
    ax1.legend(frameon=True, fontsize=8)
    
    bars = ax2.bar(cats, ratios, color="#27ae60", edgecolor="#222222", width=0.5)
    for b in bars:
        h = b.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
                     
    ax2.set_ylabel("Compression Ratio (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Tokens Saved Ratio (%)", fontsize=11, fontweight="bold")
    ax2.set_xticks(range(len(cats)))
    ax2.set_xticklabels(cats, rotation=25, ha="right", fontsize=8)
    ax2.set_ylim(0, 100)
    ax2.grid(True, axis="y")
    
    fig.suptitle("Figure 7: Context Compression Token Savings across Domains", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def chart_8_recall_at_k_curves(recall_curves: Dict[str, List[float]], output_path: Path):
    """Chart 8: Recall@K Curves (K = 1 to 10)."""
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    k_range = list(range(1, 11))
    colors = {"BM25": "#3498db", "Dense": "#e67e22", "Hybrid RRF": "#2ecc71", "Hybrid + Rerank": "#9b59b6", "Advanced RAG": "#e74c3c"}
    markers = {"BM25": "o", "Dense": "s", "Hybrid RRF": "^", "Hybrid + Rerank": "D", "Advanced RAG": "P"}
    
    for name, curve in recall_curves.items():
        c = colors.get(name, "#333333")
        m = markers.get(name, "o")
        ax.plot(k_range, curve, marker=m, linewidth=2, label=name, color=c)
        
    ax.set_xlabel("Top-K Candidates Inspected", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Recall@K", fontsize=11, fontweight="bold")
    ax.set_title("Figure 8: Recall@K Cumulative Coverage Curves (K=1 to 10)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(k_range)
    ax.set_ylim(0, 1.05)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def chart_9_mrr_by_query_type(type_data: Dict[str, Dict[str, float]], output_path: Path):
    """Chart 9: MRR Breakdown by Query Type (Lexical vs Semantic vs Mixed)."""
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    types = ["lexical", "semantic", "mixed"]
    type_labels = ["Lexical Queries (Keywords/Codes)", "Semantic Queries (Conceptual)", "Mixed Queries (Conversational)"]
    models = ["BM25", "Dense", "Hybrid RRF", "Advanced RAG"]
    model_keys = ["bm25", "dense", "hybrid_rrf", "advanced_full"]
    
    x = np.arange(len(types))
    w = 0.18
    colors = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c"]
    
    for i, (m_name, m_key, col) in enumerate(zip(models, model_keys, colors)):
        mrrs = [type_data.get(m_key, {}).get(t, {}).get("mean_mrr", 0.0) for t in types]
        offset = (i - 1.5) * w
        rects = ax.bar(x + offset, mrrs, w, label=m_name, color=col, edgecolor="#222222")
        for r in rects:
            h = r.get_height()
            if h > 0:
                ax.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7, fontweight="bold")
                            
    ax.set_ylabel("Mean Reciprocal Rank (MRR)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 9: Retrieval Quality (MRR) by Query Phrasing Type", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(type_labels, fontsize=8, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y")
    ax.legend(frameon=True, facecolor="white", loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def chart_10_metadata_filtering_performance(filter_perf: Dict[str, Any], output_path: Path):
    """Chart 10: Metadata Pre-Filtering Space Reduction & Latency."""
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    categories = list(filter_perf.keys())
    chunks_before = [filter_perf[c]["total_chunks"] for c in categories]
    chunks_after = [filter_perf[c]["filtered_chunks"] for c in categories]
    speedup = [filter_perf[c]["speedup_percent"] for c in categories]
    
    x = np.arange(len(categories))
    w = 0.35
    
    ax.bar(x - w/2, chunks_before, w, label="Unfiltered Total Chunks", color="#bdc3c7", edgecolor="#222222")
    bars2 = ax.bar(x + w/2, chunks_after, w, label="Pre-Filtered Candidate Chunks", color="#3498db", edgecolor="#222222")
    
    for i, b in enumerate(bars2):
        sp = speedup[i]
        ax.annotate(f"-{100 - (chunks_after[i]/chunks_before[i]*100):.0f}% search space",
                    xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
                    
    ax.set_ylabel("Number of Chunks Searched", fontsize=11, fontweight="bold")
    ax.set_title("Figure 10: Metadata Pre-Filtering Search Space Reduction", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").title() for c in categories], fontsize=8, fontweight="bold")
    ax.grid(True, axis="y")
    ax.legend(frameon=True, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def chart_11_error_taxonomy_distribution(error_counts: Dict[str, int], output_path: Path):
    """Chart 11: Error Taxonomy Distribution."""
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    labels = [
        "Correct Retrieval",
        "Lexical Mismatch (Dense found, BM25 missed)",
        "Semantic Confusion (BM25 found, Dense missed)",
        "Rank Decay (> top-5 cutoff)",
        "Correct Rejection (Unanswerable)",
        "Candidate Miss"
    ]
    key_map = {
        "correct_retrieval": "Correct Retrieval",
        "lexical_mismatch": "Lexical Mismatch (Dense found, BM25 missed)",
        "semantic_confusion": "Semantic Confusion (BM25 found, Dense missed)",
        "rank_decay": "Rank Decay (> top-5 cutoff)",
        "correct_rejection": "Correct Rejection (Unanswerable)",
        "candidate_miss": "Candidate Miss"
    }
    
    vals = [error_counts.get(k, 0) for k in key_map.keys()]
    clean_labels = [key_map[k] for k in key_map.keys()]
    colors = ["#2ecc71", "#3498db", "#e67e22", "#f39c12", "#1abc9c", "#e74c3c"]
    
    bars = ax.barh(clean_labels, vals, color=colors, edgecolor="#222222")
    for b in bars:
        w = b.get_width()
        if w > 0:
            ax.annotate(f" {int(w)} queries", xy=(w, b.get_y() + b.get_height() / 2),
                        ha="left", va="center", fontsize=8, fontweight="bold")
                        
    ax.set_xlabel("Number of Benchmark Queries", fontsize=11, fontweight="bold")
    ax.set_title("Figure 11: Retrieval Failure Mode & Error Taxonomy Breakdown", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, axis="x")
    ax.set_xlim(0, max(vals) * 1.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def chart_12_reranker_rank_shifts(rank_shifts: List[Dict[str, int]], output_path: Path):
    """Chart 12: Rank Shifts Before vs After Cross-Encoder Reranking."""
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    
    before_ranks = [d["before"] for d in rank_shifts]
    after_ranks = [d["after"] for d in rank_shifts]
    
    # Scatter plot
    ax.scatter(before_ranks, after_ranks, s=80, alpha=0.7, color="#8e44ad", edgecolors="#2c3e50")
    
    # Diagonal line (no shift)
    max_r = max(max(before_ranks, default=10), max(after_ranks, default=10))
    ax.plot([1, max_r], [1, max_r], color="#7f8c8d", linestyle="--", linewidth=1.5, label="No Rank Shift (Diagonal)")
    
    # Fill region of improvement
    ax.fill_between([1, max_r], [1, 1], [1, max_r], color="#2ecc71", alpha=0.1, label="Rank Improved by Reranker")
    ax.fill_between([1, max_r], [1, max_r], [max_r, max_r], color="#e74c3c", alpha=0.1, label="Rank Demoted")
    
    ax.set_xlabel("Initial Candidate Rank (Hybrid RRF)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Final Rank (After Cross-Encoder)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 12: Cross-Encoder Rank Calibration & Position Shifts", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0.5, max_r + 0.5)
    ax.set_ylim(0.5, max_r + 0.5)
    ax.invert_yaxis()
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def chart_13_chunk_diversity_index(diversity_data: Dict[str, float], output_path: Path):
    """Chart 13: Pairwise Chunk Similarity / Diversity Before vs After MMR."""
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    
    states = ["Before MMR\n(Greedy Top-K)", "After MMR\n(lambda=0.7)"]
    sim_scores = [diversity_data["before_sim"], diversity_data["after_sim"]]
    div_scores = [1.0 - diversity_data["before_sim"], 1.0 - diversity_data["after_sim"]]
    
    w = 0.4
    x = np.arange(len(states))
    
    r1 = ax.bar(x - w/2, sim_scores, w, label="Inter-Chunk Redundancy (Similarity)", color="#e74c3c", edgecolor="#222222")
    r2 = ax.bar(x + w/2, div_scores, w, label="Information Novelty (Diversity)", color="#2ecc71", edgecolor="#222222")
    
    for r in list(r1) + list(r2):
        h = r.get_height()
        ax.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
                    
    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("Figure 13: Inter-Chunk Redundancy Reduction via MMR", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(states, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y")
    ax.legend(frameon=True, facecolor="white", loc="upper center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def chart_14_end_to_end_radar(radar_metrics: Dict[str, Dict[str, float]], output_path: Path):
    """Chart 14: Basic RAG vs Advanced RAG End-to-End Comparison across 5 IR Pillars."""
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    dimensions = ["Precision@5", "Recall@5", "MRR", "nDCG@5", "Grounded Faithfulness"]
    keys = ["p@5", "r@5", "mrr", "ndcg@5", "faithfulness"]
    
    basic_vals = [radar_metrics["basic"].get(k, 0.0) for k in keys]
    advanced_vals = [radar_metrics["advanced"].get(k, 0.0) for k in keys]
    
    x = np.arange(len(dimensions))
    w = 0.35
    
    r1 = ax.bar(x - w/2, basic_vals, w, label="Basic RAG (Dense Top-K, Uncompressed)", color="#95a5a6", edgecolor="#222222")
    r2 = ax.bar(x + w/2, advanced_vals, w, label="Advanced RAG (Hybrid + RRF + Rerank + MMR + Compression)", color="#e74c3c", edgecolor="#222222")
    
    for r in list(r1) + list(r2):
        h = r.get_height()
        ax.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
                    
    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("Figure 14: Basic RAG vs Advanced RAG Comprehensive Architecture Comparison", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.2)
    ax.grid(True, axis="y")
    ax.legend(frameon=True, facecolor="white", loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def chart_15_knowledge_base_distribution(kb_stats: Dict[str, Any], output_path: Path):
    """Chart 15: Knowledge Base Category Breakdown and Chunk Length Distribution."""
    setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    
    # Panel 1: Document distribution
    categories = list(kb_stats["category_counts"].keys())
    counts = list(kb_stats["category_counts"].values())
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c", "#1abc9c"]
    
    bars = ax1.bar([c.replace("_", " ").title() for c in categories], counts, color=colors[:len(categories)], edgecolor="#222222")
    for b in bars:
        ax1.annotate(f"{int(b.get_height())} docs", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
                     
    ax1.set_ylabel("Document Count", fontsize=10, fontweight="bold")
    ax1.set_title("Corpus Category Breakdown", fontsize=11, fontweight="bold")
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels([c.replace("_", " ").title() for c in categories], rotation=25, ha="right", fontsize=8)
    ax1.grid(True, axis="y")
    
    # Panel 2: Chunk token length histogram
    lengths = kb_stats["chunk_token_lengths"]
    ax2.hist(lengths, bins=12, color="#34495e", edgecolor="#ffffff", alpha=0.85)
    mean_len = float(np.mean(lengths))
    ax2.axvline(mean_len, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Mean: {mean_len:.1f} tokens")
    
    ax2.set_xlabel("Token Count per Chunk", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Chunk Frequency", fontsize=10, fontweight="bold")
    ax2.set_title("Chunk Token Length Distribution", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="y")
    ax2.legend(frameon=True, fontsize=8)
    
    fig.suptitle(f"Figure 15: Synthetic Knowledge Base Characteristics (N={len(lengths)} Chunks)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
