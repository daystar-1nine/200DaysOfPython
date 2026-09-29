"""
Dataset splitting module with stratified train/validation/test partitioning.
"""
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs stratified 3-way partition: train, validation, test.
    Ensures zero leakage across evaluation splits.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df["label"],
        random_state=random_seed
    )

    val_relative_size = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        stratify=train_val_df["label"],
        random_state=random_seed
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )
