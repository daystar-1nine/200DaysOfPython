from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "sms_spam.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "output"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
CHARTS_DIR = OUTPUT_DIR / "charts"
MODELS_DIR = BASE_DIR / "models"

for dir_path in [PROCESSED_DATA_DIR, PREDICTIONS_DIR, CHARTS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Global hyperparameters & reproducibility
SEED = 42
MAX_VOCAB_SIZE = 5000
MAX_SEQ_LENGTH = 50
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
DENSE_DIM = 32
DROPOUT_RATE = 0.3
DENSE_DROPOUT_RATE = 0.2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1

# Hardware device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
