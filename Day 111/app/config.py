"""
Configuration settings for Day 111: BERT & Bidirectional Transformers.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Base Directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    CHARTS_DIR: Path = BASE_DIR / "output" / "charts"
    EMBEDDINGS_DIR: Path = BASE_DIR / "output" / "embeddings"
    MODELS_DIR: Path = BASE_DIR / "models"

    RAW_DATA_FILE: Path = DATA_RAW_DIR / "sms_spam.csv"

    # Preprocessing
    MAX_LENGTH: int = 128
    BERT_MODEL_NAME: str = "prajjwal1/bert-tiny"

    # Dataset Split Ratios
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    RANDOM_SEED: int = 42

    # Training Parameters
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 2e-5
    FROZEN_LR: float = 1e-3
    FINETUNED_LR: float = 2e-5
    EPOCHS: int = 3
    PATIENCE: int = 3
    DROPOUT: float = 0.1

    # Evaluation Thresholds
    THRESHOLDS: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    def __post_init__(self):
        self.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        self.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)


config = Config()

# Module-level aliases
RAW_DATA_PATH = config.RAW_DATA_FILE
OUTPUT_DIR = config.OUTPUT_DIR
CHARTS_DIR = config.CHARTS_DIR
MODEL_NAME = config.BERT_MODEL_NAME
MAX_LENGTH = config.MAX_LENGTH
BATCH_SIZE = config.BATCH_SIZE
EPOCHS = config.EPOCHS
FROZEN_LR = config.FROZEN_LR
FINETUNED_LR = config.FINETUNED_LR
