"""
Configuration module for Day 105: Neural NLP & Text Classification.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration settings for neural text classification pipeline."""

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    CHARTS_DIR: Path = OUTPUT_DIR / "charts"
    MODELS_DIR: Path = OUTPUT_DIR / "models"
    EMBEDDINGS_DIR: Path = OUTPUT_DIR / "embeddings"

    # File Paths
    RAW_DATA_PATH: Path = RAW_DATA_DIR / "sms_spam.csv"
    TRAIN_DATA_PATH: Path = PROCESSED_DATA_DIR / "train.csv"
    VAL_DATA_PATH: Path = PROCESSED_DATA_DIR / "val.csv"
    TEST_DATA_PATH: Path = PROCESSED_DATA_DIR / "test.csv"
    BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.pt"
    PREDICTIONS_PATH: Path = OUTPUT_DIR / "predictions.csv"
    METRICS_PATH: Path = OUTPUT_DIR / "metrics.csv"
    ERRORS_PATH: Path = OUTPUT_DIR / "errors.csv"
    REPORT_PATH: Path = OUTPUT_DIR / "report.md"

    # Special Tokens
    PAD_TOKEN: str = "<PAD>"
    UNK_TOKEN: str = "<UNK>"
    PAD_ID: int = 0
    UNK_ID: int = 1

    # Preprocessing
    MIN_WORD_FREQ: int = 2
    MAX_SEQ_LEN: int = 40
    TRUNCATION_STRATEGY: str = "post"  # 'post' or 'pre'
    PADDING_STRATEGY: str = "post"     # 'post' or 'pre'

    # Model Hyperparameters
    EMBEDDING_DIM: int = 64
    HIDDEN_DIM: int = 64
    DROPOUT_RATE: float = 0.5

    # Training Hyperparameters
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.002
    EPOCHS: int = 25
    PATIENCE: int = 5
    RANDOM_SEED: int = 42

    # Data Splits
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    def ensure_directories(self) -> None:
        """Create output and data directories if they do not exist."""
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


config = ModelConfig()
