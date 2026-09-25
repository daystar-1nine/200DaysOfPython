"""
Configuration settings and hyperparameters for Day 106: RNNs & Sequential Text Learning.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"
MODELS_DIR = BASE_DIR / "models"

RAW_DATA_PATH = DATA_RAW_DIR / "sms_spam.csv"
TRAIN_DATA_PATH = DATA_PROCESSED_DIR / "train.csv"
VAL_DATA_PATH = DATA_PROCESSED_DIR / "val.csv"
TEST_DATA_PATH = DATA_PROCESSED_DIR / "test.csv"
VOCAB_PATH = DATA_PROCESSED_DIR / "vocabulary.json"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.json"
MODEL_CHECKPOINT_PATH = MODELS_DIR / "best_rnn.pt"

# Reproducibility
RANDOM_SEED = 42

# Special Tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1
MIN_WORD_FREQ = 2

# Model Hyperparameters (Default Architecture B)
EMBEDDING_DIM = 64
HIDDEN_UNITS = 64
DENSE_UNITS = 32
MAX_SEQ_LENGTH = 40
DROPOUT_1 = 0.3
DROPOUT_2 = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 25
PATIENCE = 5
MIN_DELTA = 1e-4

# Thresholds to evaluate
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
DEFAULT_THRESHOLD = 0.50
