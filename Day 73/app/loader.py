"""
Day 73 - Dataset Loader
Reads CSV data with existence, path, and non-empty validation.
"""
import os
import pandas as pd

def load_csv(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError(f"Dataset at {filepath} is empty.")
        
    return df
