"""
Dataset splitting module for Day 106: RNNs & Sequential Text Learning.
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
            df: Input DataFrame with 'label' or 'target' column.
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

    @classmethod
    def split_and_save(
        cls,
        df: pd.DataFrame,
        train_path: Union[str, Path],
        val_path: Union[str, Path],
        test_path: Union[str, Path],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split DataFrame and persist subsets as CSV files."""
        train_df, val_df, test_df = cls.split(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed
        )

        Path(train_path).parent.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        return train_df, val_df, test_df
