import os
import pandas as pd

def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw customer churn dataset from CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at path: {file_path}")
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"Dataset at {file_path} is empty.")
    return df
