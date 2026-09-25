"""
Data loading module for Day 106: RNNs & Sequential Text Learning.
"""

from pathlib import Path
from typing import Union
import pandas as pd


class DataLoader:
    """Loads and standardizes raw dataset files."""

    @staticmethod
    def load_csv(file_path: Union[str, Path]) -> pd.DataFrame:
        """Load CSV dataset into DataFrame.
        
        Args:
            file_path: Path to the raw CSV file.
            
        Returns:
            Cleaned pandas DataFrame with 'label' and 'text' columns.
            
        Raises:
            FileNotFoundError: If file_path does not exist.
            ValueError: If file is empty or cannot be parsed.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path.resolve()}")

        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")

        if df.empty:
            raise ValueError(f"Dataset at {path.resolve()} is empty.")

        # Standardize column names
        df.columns = [c.strip().lower() for c in df.columns]

        # Handle 'message' vs 'text' aliases if needed
        if "message" in df.columns and "text" not in df.columns:
            df.rename(columns={"message": "text"}, inplace=True)

        if "v1" in df.columns and "v2" in df.columns:
            df.rename(columns={"v1": "label", "v2": "text"}, inplace=True)

        # Drop null values
        df = df.dropna(subset=["label", "text"]).copy()
        df["text"] = df["text"].astype(str).str.strip()
        df["label"] = df["label"].astype(str).str.strip().str.lower()

        # Filter out empty text entries
        df = df[df["text"].str.len() > 0].reset_index(drop=True)

        # Map binary target: ham -> 0, spam -> 1
        df["target"] = (df["label"] == "spam").astype(int)

        return df
