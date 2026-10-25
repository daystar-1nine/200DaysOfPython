"""
Reporting and visualization package.
"""
from app.reporting.charts import (
    chart_1_bm25_vs_dense_vs_hybrid,
    chart_2_weighted_vs_rrf,
    chart_3_query_optimization_ablation,
    chart_4_reranking_ndcg_gain,
    chart_5_mmr_diversity_frontier,
    chart_6_latency_vs_mrr_tradeoff,
    chart_7_context_compression_efficiency,
    chart_8_recall_at_k_curves,
    chart_9_mrr_by_query_type,
    chart_10_metadata_filtering_performance,
    chart_11_error_taxonomy_distribution,
    chart_12_reranker_rank_shifts,
    chart_13_chunk_diversity_index,
    chart_14_end_to_end_radar,
    chart_15_knowledge_base_distribution,
)

__all__ = [
    "chart_1_bm25_vs_dense_vs_hybrid",
    "chart_2_weighted_vs_rrf",
    "chart_3_query_optimization_ablation",
    "chart_4_reranking_ndcg_gain",
    "chart_5_mmr_diversity_frontier",
    "chart_6_latency_vs_mrr_tradeoff",
    "chart_7_context_compression_efficiency",
    "chart_8_recall_at_k_curves",
    "chart_9_mrr_by_query_type",
    "chart_10_metadata_filtering_performance",
    "chart_11_error_taxonomy_distribution",
    "chart_12_reranker_rank_shifts",
    "chart_13_chunk_diversity_index",
    "chart_14_end_to_end_radar",
    "chart_15_knowledge_base_distribution",
]
