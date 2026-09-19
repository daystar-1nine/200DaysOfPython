"""Data loading module for text datasets."""
from pathlib import Path
import pandas as pd
from app.data.generate_dataset import generate_sms_dataset

def load_sms_data(filepath: Path) -> pd.DataFrame:
    """Loads SMS Spam dataset from CSV, generating it if not found."""
    if not filepath.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)
        return generate_sms_dataset(str(filepath), n_samples=800, seed=42)
    
    df = pd.read_csv(filepath, encoding="utf-8")
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("Dataset must contain 'label' and 'text' columns.")
    return df
