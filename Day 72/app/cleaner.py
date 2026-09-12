"""
Day 72 — Dataset Cleaner
Cleans dataset by isolating numerical columns, removing infinities, and dropping null values.
"""
import numpy as np
import pandas as pd

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Filter only numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numerical columns found in the dataset.")
        
    cleaned = df[numeric_cols].copy()
    
    # 2. Replace inf and -inf with NaN
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    
    # 3. Drop rows with any NaN values
    cleaned = cleaned.dropna()
    
    if cleaned.empty:
        raise ValueError("All rows were dropped during cleaning due to NaN/Inf values.")
        
    return cleaned
