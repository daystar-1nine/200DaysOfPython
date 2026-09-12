import os
import pandas as pd
from typing import Union
from app.config import AppConfig

def load_data(config_or_path: Union[AppConfig, str] = None) -> pd.DataFrame:
    if config_or_path is None:
        config = AppConfig()
        path = config.DATA_PATH_RAW
    elif isinstance(config_or_path, AppConfig):
        path = config_or_path.DATA_PATH_RAW
    else:
        path = config_or_path
        
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")
        
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Data file is empty")
        
    return df
