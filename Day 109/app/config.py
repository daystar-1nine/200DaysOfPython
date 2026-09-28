from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "sms_spam.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"
ATTENTION_EXAMPLES_DIR = OUTPUT_DIR / "attention_examples"
MODELS_DIR = BASE_DIR / "models"

for dir_path in [PROCESSED_DATA_DIR, OUTPUT_DIR, CHARTS_DIR, ATTENTION_EXAMPLES_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Hyperparameters & Reproducibility
SEED = 42
MAX_VOCAB_SIZE = 5000
MAX_SEQ_LENGTH = 50
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
ATTENTION_DIM = 64
DENSE_DIM = 32
DROPOUT_RATE = 0.3
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
