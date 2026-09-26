from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "sms_spam.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"
MODELS_DIR = OUTPUT_DIR / "models"

# Ensure directories exist
for d in [DATA_DIR, RAW_DATA_PATH.parent, PROCESSED_DATA_DIR, OUTPUT_DIR, CHARTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset Config
TEXT_COL = "text"
LABEL_COL = "label"
MAX_SEQ_LEN_DEFAULT = 40

# Training Config
BATCH_SIZE = 32
RANDOM_SEED = 42

# Vocabulary Config
MAX_VOCAB_SIZE = 10000
OOV_TOKEN = "<OOV>"
PAD_TOKEN = "<PAD>"
PAD_ID = 0
OOV_ID = 1
