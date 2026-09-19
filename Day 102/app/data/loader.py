"""Data loading utilities."""
from pathlib import Path
import pandas as pd

def load_data(filepath: Path) -> pd.DataFrame:
    """Loads dataset from CSV file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath, encoding="utf-8")
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("Dataset must contain 'label' and 'text' columns.")
    return df
