"""
Data loader module for Day 105: Neural NLP & Text Classification.
"""

from pathlib import Path
from typing import Tuple, Union
import pandas as pd


class DataLoader:
    """Loads and standardizes SMS Spam text dataset."""

    @staticmethod
    def load_raw_data(filepath: Union[str, Path]) -> pd.DataFrame:
        """Load raw CSV dataset.
        
        Args:
            filepath: Path to sms_spam.csv.
            
        Returns:
            DataFrame with 'label' and 'text' columns.
            
        Raises:
            FileNotFoundError: If filepath does not exist.
            ValueError: If required columns are missing.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found at: {path}")

        df = pd.read_csv(path)
        required_cols = {"label", "text"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Dataset missing required columns {required_cols}. Found: {list(df.columns)}")

        # Clean string whitespace and ensure proper types
        df["text"] = df["text"].astype(str).str.strip()
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        return df

    @staticmethod
    def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
        """Map categorical 'ham'/'spam' labels to binary integers (ham=0, spam=1).
        
        Args:
            df: DataFrame containing 'label' column.
            
        Returns:
            DataFrame with added 'target' integer column.
        """
        df_copy = df.copy()
        label_map = {"ham": 0, "spam": 1}
        invalid_labels = set(df_copy["label"].unique()) - set(label_map.keys())
        if invalid_labels:
            raise ValueError(f"Encountered unexpected labels: {invalid_labels}")

        df_copy["target"] = df_copy["label"].map(label_map).astype(int)
        return df_copy
