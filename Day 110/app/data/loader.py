"""
Data loader for SMS Spam dataset.
"""
from pathlib import Path
import pandas as pd


def load_raw_sms_data(file_path: Path) -> pd.DataFrame:
    """Loads raw SMS dataset handling encodings."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not decode file {path} with supported encodings.")
