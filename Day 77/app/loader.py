import os
import pandas as pd
from app.config import AppConfig

def load_data(file_path: str = None, config: AppConfig = None) -> pd.DataFrame:
    if config is None:
        config = AppConfig()
    if file_path is None:
        file_path = config.DATA_PATH_RAW
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
        
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"Loaded dataset from {file_path} is empty.")
    return df
