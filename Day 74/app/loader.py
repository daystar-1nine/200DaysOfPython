import os
import pandas as pd

try:
    from app.config import AppConfig
except ImportError:
    from config import AppConfig

from typing import Union

def load_data(config_or_path: Union[AppConfig, str, None] = None) -> pd.DataFrame:
    """
    Reads CSV from DATA_PATH_RAW or specified file path.
    Validates file exists and required columns are present.
    """
    if isinstance(config_or_path, str):
        path = config_or_path
        config = AppConfig()
    elif isinstance(config_or_path, AppConfig):
        config = config_or_path
        path = config.DATA_PATH_RAW
    else:
        config = AppConfig()
        path = config.DATA_PATH_RAW

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")

    df = pd.read_csv(path)

    required_cols = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES + [config.TARGET_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    return df
