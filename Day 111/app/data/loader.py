"""
Data loader module for loading raw SMS Spam dataset.
"""
from pathlib import Path
import pandas as pd


def load_raw_sms_data(file_path: Path) -> pd.DataFrame:
    """Loads raw SMS dataset with multi-encoding support."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Failed to read {path} with supported encodings.")
