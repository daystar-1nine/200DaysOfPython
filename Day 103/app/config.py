"""Configuration settings for Day 103 Word2Vec Engine."""
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_RAW: Path = BASE_DIR / "data" / "raw" / "corpus.txt"
    DATA_PROCESSED: Path = BASE_DIR / "data" / "processed" / "tokens.json"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    CHARTS_DIR: Path = BASE_DIR / "output" / "charts"
    EMBEDDINGS_PATH: Path = BASE_DIR / "output" / "embeddings.npy"
    VOCAB_PATH: Path = BASE_DIR / "output" / "vocabulary.json"
    SIMILARITY_MATRIX_PATH: Path = BASE_DIR / "output" / "similarity_matrix.csv"
    NEAREST_NEIGHBORS_PATH: Path = BASE_DIR / "output" / "nearest_neighbors.csv"
    TRAINING_LOSS_PATH: Path = BASE_DIR / "output" / "training_loss.csv"
    REPORT_PATH: Path = BASE_DIR / "output" / "report.md"
    
    # Model defaults
    EMBEDDING_DIM: int = 16
    WINDOW_SIZE: int = 2
    NEGATIVE_SAMPLES: int = 4
    LEARNING_RATE: float = 0.025
    EPOCHS: int = 50
    RANDOM_STATE: int = 42

config = Config()
