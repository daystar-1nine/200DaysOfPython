import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from app.config import SEED

def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (0.999 <= train_ratio + val_ratio + test_ratio <= 1.001):
        raise ValueError("Ratios must sum to 1.0")
        
    test_and_val_size = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=test_and_val_size,
        stratify=df["label"],
        random_state=seed
    )
    
    relative_test_size = test_ratio / test_and_val_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["label"],
        random_state=seed
    )
    
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )
