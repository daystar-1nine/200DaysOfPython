"""
Global configuration for Day 117 Mini RAG system.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any


@dataclass
class RAGConfig:
    """Configuration settings for document ingestion, chunking, retrieval, and generation."""
    # Chunking parameters
    chunk_size: int = 400
    chunk_overlap: int = 50
    chunking_strategy: str = "sentence"  # "character", "word", "sentence"
    
    # Embedding & Index parameters
    embedding_dim: int = 64
    embedding_type: str = "dense_tfidf"  # "dense_tfidf", "sparse_tfidf", "svd"
    similarity_metric: str = "cosine"
    
    # Retrieval parameters
    top_k: int = 3
    retrieval_method: str = "hybrid"  # "keyword", "semantic", "hybrid"
    hybrid_alpha: float = 0.5         # 0.0 = pure keyword, 1.0 = pure semantic
    min_score_threshold: float = 0.05
    
    # Generation parameters
    max_context_length: int = 1500
    enable_anti_hallucination: bool = True
    include_citations: bool = True
    
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
