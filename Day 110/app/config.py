"""
Global configuration for Day 110: Transformers From Scratch.
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
    ATTENTION_DIR: Path = BASE_DIR / "output" / "transformer_attention"
    MODELS_DIR: Path = BASE_DIR / "models"

    RAW_DATA_FILE: Path = DATA_RAW_DIR / "sms_spam.csv"

    # Preprocessing
    MAX_LENGTH: int = 100
    MIN_FREQ: int = 2
    PAD_TOKEN: str = "<PAD>"
    UNK_TOKEN: str = "<UNK>"
    PAD_IDX: int = 0
    UNK_IDX: int = 1

    # Split Ratios
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    RANDOM_SEED: int = 42

    # Transformer Architecture Defaults
    D_MODEL: int = 128
    NUM_HEADS: int = 4
    D_FF: int = 256
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.1

    # Training
    BATCH_SIZE: int = 32
    EPOCHS: int = 15
    LEARNING_RATE: float = 1e-3
    PATIENCE: int = 4

    # Experiments
    EXPERIMENT_HEADS: List[int] = field(default_factory=lambda: [2, 4, 8])
    EXPERIMENT_LAYERS: List[int] = field(default_factory=lambda: [1, 2, 4])
    EXPERIMENT_DIMS: List[int] = field(default_factory=lambda: [64, 128, 256])
    EXPERIMENT_SEQS: List[int] = field(default_factory=lambda: [32, 64, 100, 150])

    def __post_init__(self):
        self.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        self.ATTENTION_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)


config = Config()
