"""
Day 72 — Dataset Loader
Handles ingestion of CSV and JSON formatted datasets with validation.
"""
import os
import pandas as pd

def load_dataset(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at path: {filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext == ".json":
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Expected .csv or .json.")
        
    if df.empty:
        raise ValueError(f"Dataset at {filepath} is empty.")
        
    return df
