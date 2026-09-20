"""
Configuration module for Day 104 Semantic Search Engine.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SearchConfig:
    """Configuration settings for semantic and keyword search pipelines."""
    
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    EVALUATION_DATA_DIR: Path = DATA_DIR / "evaluation"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    CHARTS_DIR: Path = OUTPUT_DIR / "charts"
    
    # File paths
    DOCUMENTS_PATH: Path = RAW_DATA_DIR / "documents.json"
    PROCESSED_DOCS_PATH: Path = PROCESSED_DATA_DIR / "processed_docs.json"
    GROUND_TRUTH_PATH: Path = EVALUATION_DATA_DIR / "ground_truth.json"
    EMBEDDINGS_PATH: Path = OUTPUT_DIR / "embeddings.npy"
    SEARCH_RESULTS_PATH: Path = OUTPUT_DIR / "search_results.csv"
    RETRIEVAL_METRICS_PATH: Path = OUTPUT_DIR / "retrieval_metrics.csv"
    ERRORS_PATH: Path = OUTPUT_DIR / "errors.csv"
    REPORT_PATH: Path = OUTPUT_DIR / "report.md"
    
    # Preprocessing parameters
    LOWERCASE: bool = True
    REMOVE_PUNCTUATION: bool = True
    STRIP_WHITESPACE: bool = True
    MIN_TOKEN_LEN: int = 2
    
    # Representation parameters
    EMBEDDING_DIM: int = 64
    UNK_STRATEGY: str = "zero"  # "zero" or "random"
    DEFAULT_POOLING: str = "mean"  # "mean" or "weighted"
    
    # Retrieval parameters
    DEFAULT_TOP_K: int = 5
    HYBRID_ALPHA: float = 0.5  # Weight for TF-IDF in hybrid search (1 - alpha for embeddings)
    
    # Evaluation parameters
    EVAL_K_VALUES: tuple = (1, 3, 5)
    
    # Reproducibility
    RANDOM_SEED: int = 42

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.EVALUATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CHARTS_DIR.mkdir(parents=True, exist_ok=True)


config = SearchConfig()
