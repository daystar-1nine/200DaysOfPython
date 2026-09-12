import os
import pandas as pd
from typing import Optional
from app.config import AppConfig

def load_data(config_or_path: Optional[str] = None) -> pd.DataFrame:
    if isinstance(config_or_path, str):
        path = config_or_path
    else:
        path = AppConfig().DATA_PATH_RAW
        
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")
        
    df = pd.read_csv(path)
    required_cols = AppConfig().NUMERIC_FEATURES + AppConfig().CATEGORICAL_FEATURES + [AppConfig().TARGET_COL]
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
        
    return df
