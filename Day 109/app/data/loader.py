import pandas as pd
from pathlib import Path
from typing import Union

def load_raw_data(file_path: Union[str, Path]) -> pd.DataFrame:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1")
        
    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]].copy()
        df.columns = ["label", "text"]
    elif "label" in df.columns and "text" in df.columns:
        df = df[["label", "text"]].copy()
    else:
        df = df.iloc[:, :2].copy()
        df.columns = ["label", "text"]
        
    return df
