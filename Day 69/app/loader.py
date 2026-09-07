"""
Resilient data loader for CSV, JSON, and in-memory structures.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
try:
    from app.validator import validate_numeric_array
except (ImportError, ModuleNotFoundError):
    from validator import validate_numeric_array

class DataLoader:
    @staticmethod
    def load_from_csv(file_path: str | Path, column: str = "order_value") -> np.ndarray:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"CSV file not found: {p}")
        df = pd.read_csv(p)
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in CSV. Available: {list(df.columns)}")
        clean_vals = df[column].dropna().values
        return validate_numeric_array(clean_vals)

    @staticmethod
    def load_from_json(file_path: str | Path, key: str | None = None) -> np.ndarray:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"JSON file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if key is not None and isinstance(data, dict):
            if key not in data:
                raise KeyError(f"Key '{key}' not found in JSON object.")
            data = data[key]
        return validate_numeric_array(data)

    @staticmethod
    def load_from_list(data: list[float] | np.ndarray) -> np.ndarray:
        return validate_numeric_array(data)
