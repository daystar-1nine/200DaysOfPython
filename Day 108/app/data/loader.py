import pandas as pd
from pathlib import Path
from typing import Union

def load_raw_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads raw SMS spam dataset handling encoding issues (latin-1 fallback).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1")
    
    # SMS Spam collection typically has columns v1 (label) and v2 (text) or label/text
    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]].copy()
        df.columns = ["label", "text"]
    elif "label" in df.columns and "text" in df.columns:
        df = df[["label", "text"]].copy()
    else:
        # Take first two columns if column names differ
        df = df.iloc[:, :2].copy()
        df.columns = ["label", "text"]
        
    return df
