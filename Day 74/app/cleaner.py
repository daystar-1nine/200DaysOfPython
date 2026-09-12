import os
import pandas as pd

try:
    from app.config import AppConfig
except ImportError:
    from config import AppConfig

from typing import Optional

def clean_data(df: pd.DataFrame, config: Optional[AppConfig] = None) -> pd.DataFrame:
    """
    Cleans the raw DataFrame according to business rules.
    """
    if config is None:
        config = AppConfig()

    df = df.copy()

    # Drop rows where Sales is null or <= 0
    if config.TARGET_COL in df.columns:
        df = df.dropna(subset=[config.TARGET_COL])
        df = df[df[config.TARGET_COL] > 0]

    # Fill NaN TV_Spend with median, then drop TV_Spend < 0
    if "TV_Spend" in df.columns:
        median_tv = df["TV_Spend"].median()
        if pd.notna(median_tv):
            df["TV_Spend"] = df["TV_Spend"].fillna(median_tv)
        df = df[df["TV_Spend"] >= 0]

    # Clip Discount to [0, 100]
    if "Discount" in df.columns:
        df["Discount"] = df["Discount"].clip(lower=0, upper=100)

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Drop rows where Region == "Unknown" or not in VALID_REGIONS
    if "Region" in df.columns:
        df = df[df["Region"].isin(config.VALID_REGIONS)]

    # Drop rows where Category not in VALID_CATEGORIES
    if "Category" in df.columns:
        df = df[df["Category"].isin(config.VALID_CATEGORIES)]

    # Save cleaned data if path configured
    try:
        if config.DATA_PATH_PROCESSED:
            os.makedirs(os.path.dirname(config.DATA_PATH_PROCESSED), exist_ok=True)
            df.to_csv(config.DATA_PATH_PROCESSED, index=False)
    except Exception:
        pass

    return df
