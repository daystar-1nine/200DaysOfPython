"""
Retrieval package for sparse, dense, hybrid search, RRF, metadata filtering, query optimization, reranking, and MMR.
"""
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid import HybridRetriever, min_max_normalize
from app.retrieval.rrf import reciprocal_rank_fusion, ReciprocalRankFusion
from app.retrieval.metadata_filter import MetadataFilter
from app.retrieval.query_expansion import (
    expand_query,
    rewrite_query,
    generate_subqueries,
    QueryOptimizer,
    TECHNICAL_SYNONYMS,
)
from app.retrieval.reranker import (
    BaseReranker,
    CrossEncoderReranker,
    IdentityReranker,
)
from app.retrieval.mmr import (
    maximal_marginal_relevance,
    MaximalMarginalRelevance,
    jaccard_similarity,
)

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "min_max_normalize",
    "reciprocal_rank_fusion",
    "ReciprocalRankFusion",
    "MetadataFilter",
    "expand_query",
    "rewrite_query",
    "generate_subqueries",
    "QueryOptimizer",
    "TECHNICAL_SYNONYMS",
    "BaseReranker",
    "CrossEncoderReranker",
    "IdentityReranker",
    "maximal_marginal_relevance",
    "MaximalMarginalRelevance",
    "jaccard_similarity",
]
