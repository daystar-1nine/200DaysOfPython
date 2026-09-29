"""
Configuration settings for Day 112: MiniGPT & Autoregressive Transformers.
"""
from pathlib import Path
from dataclasses import dataclass
import torch


@dataclass
class GPTConfig:
    # Directories & Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    INPUT_FILE: Path = DATA_DIR / "input.txt"
    OUTPUTS_DIR: Path = BASE_DIR / "outputs"
    CHARTS_DIR: Path = OUTPUTS_DIR / "charts"

    # Model Hyperparameters
    CONTEXT_LENGTH: int = 64
    EMBED_DIM: int = 128
    NUM_HEADS: int = 4
    NUM_LAYERS: int = 4
    DROPOUT: float = 0.1

    # Training Parameters
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY: float = 0.01
    MAX_ITERS: int = 600
    EVAL_INTERVAL: int = 100
    EVAL_ITERS: int = 20
    VAL_RATIO: float = 0.10
    GRAD_CLIP: float = 1.0
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Generation Parameters
    PROMPTS: tuple = (
        "First Citizen:\n",
        "MENENIUS:\n",
        "MARCIUS:\n",
        "The people are ",
        "Why stay we "
    )

    def __post_init__(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        self.CHARTS_DIR.mkdir(parents=True, exist_ok=True)


config = GPTConfig()
