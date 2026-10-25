"""
Global configuration for Day 118 Advanced RAG system.
Supports Hybrid Search, RRF, Query Optimization, Cross-Encoder Reranking, and Context Compression.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any


@dataclass
class AdvancedRAGConfig:
    """Configuration settings for ingestion, chunking, retrieval, reranking, and generation."""
    # Chunking parameters
    chunk_size: int = 400
    chunk_overlap: int = 50
    chunking_strategy: str = "sentence"  # "sentence", "character", "word"
    
    # Embedding parameters
    embedding_dim: int = 64
    embedding_type: str = "dense_tfidf"
    similarity_metric: str = "cosine"
    
    # BM25 parameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_epsilon: float = 0.25
    
    # Retrieval parameters
    top_k: int = 5
    candidate_k: int = 20
    retrieval_method: str = "hybrid_rrf"  # "bm25", "dense", "hybrid_weighted", "hybrid_rrf"
    hybrid_alpha: float = 0.5              # weight for dense in weighted fusion: alpha * dense + (1 - alpha) * bm25
    rrf_k0: int = 60                       # smoothing constant in RRF: 1 / (k0 + rank)
    
    # Query expansion & rewriting parameters
    enable_query_expansion: bool = False
    enable_query_rewriting: bool = False
    enable_multi_query: bool = False
    multi_query_count: int = 3
    
    # Cross-encoder reranker parameters
    enable_reranking: bool = False
    reranker_candidates: int = 15
    
    # Maximal Marginal Relevance (MMR) parameters
    enable_mmr: bool = False
    mmr_lambda: float = 0.7  # 1.0 = pure relevance, 0.0 = pure diversity
    
    # Context compression parameters
    enable_compression: bool = False
    compression_similarity_threshold: float = 0.15
    max_context_length: int = 1200
    
    # Generation & Grounding parameters
    enable_anti_hallucination: bool = True
    include_citations: bool = True
    min_score_threshold: float = 0.01
    
    # Supported file formats
    supported_extensions: Tuple[str, ...] = (".txt", ".md", ".json", ".csv")
    
    # Standard directories
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(init=False)
    data_raw_dir: Path = field(init=False)
    data_processed_dir: Path = field(init=False)
    data_eval_dir: Path = field(init=False)
    index_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    charts_dir: Path = field(init=False)
    experiments_dir: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.base_dir / "data"
        self.data_raw_dir = self.data_dir / "raw"
        self.data_processed_dir = self.data_dir / "processed"
        self.data_eval_dir = self.data_dir / "evaluation"
        self.index_dir = self.base_dir / "index"
        self.outputs_dir = self.base_dir / "outputs"
        self.charts_dir = self.outputs_dir / "charts"
        self.experiments_dir = self.base_dir / "experiments"
        
        # Ensure directories exist
        for directory in (
            self.data_raw_dir,
            self.data_processed_dir,
            self.data_eval_dir,
            self.index_dir,
            self.charts_dir,
            self.experiments_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
