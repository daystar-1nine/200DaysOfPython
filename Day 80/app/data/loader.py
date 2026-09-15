import os
import pandas as pd

def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw dataset from CSV file with error handling."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"Dataset at {file_path} is empty.")
    return df
