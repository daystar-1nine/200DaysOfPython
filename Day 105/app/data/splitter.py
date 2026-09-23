"""
Dataset splitting module for Day 105: Neural NLP & Text Classification.
Performs stratified train/val/test splits to guarantee identical class distributions.
"""

from pathlib import Path
from typing import Tuple, Union
import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:
    """Partitions dataset into Train, Validation, and Test subsets with stratification."""

    @staticmethod
    def split(
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split DataFrame into train, val, and test partitions.
        
        Args:
            df: Input DataFrame with 'target' column.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            test_ratio: Fraction for testing.
            random_seed: Reproducibility seed.
            
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"
        stratify_col = df["target"] if "target" in df.columns else df["label"]

        # First split: train vs (val + test)
        temp_ratio = val_ratio + test_ratio
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_ratio,
            random_state=random_seed,
            stratify=stratify_col
        )

        # Second split: val vs test from temp
        val_fraction_of_temp = val_ratio / temp_ratio
        stratify_temp = temp_df["target"] if "target" in temp_df.columns else temp_df["label"]
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1.0 - val_fraction_of_temp),
            random_state=random_seed,
            stratify=stratify_temp
        )

        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True)
        )

    @staticmethod
    def save_splits(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        processed_dir: Union[str, Path]
    ) -> None:
        """Save train, val, test subsets to CSV."""
        p = Path(processed_dir)
        p.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(p / "train.csv", index=False)
        val_df.to_csv(p / "val.csv", index=False)
        test_df.to_csv(p / "test.csv", index=False)
