"""
Data loader module for Day 65 Statistical Analysis Engine.
"""

import os
import pandas as pd


def load_dataset(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError("Loaded dataset is empty.")
        
    return df
