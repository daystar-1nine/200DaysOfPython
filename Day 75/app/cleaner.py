import os
import pandas as pd
from typing import Optional
from app.config import AppConfig

def clean_data(df: pd.DataFrame, config: Optional[AppConfig] = None) -> pd.DataFrame:
    if config is None:
        config = AppConfig()
        
    # Drop null or negative targets
    df = df.dropna(subset=[config.TARGET_COL])
    df = df[df[config.TARGET_COL] > 0]
    
    # Clip discount
    df['Discount'] = df['Discount'].clip(0, 100)
    
    # Fill TV_Spend median
    if 'TV_Spend' in df.columns:
        df['TV_Spend'] = df['TV_Spend'].fillna(df['TV_Spend'].median())
        
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Filter valid regions/categories
    if 'Region' in df.columns:
        df = df[df['Region'].isin(config.VALID_REGIONS)]
    if 'Category' in df.columns:
        df = df[df['Category'].isin(config.VALID_CATEGORIES)]
        
    os.makedirs(os.path.dirname(config.DATA_PATH_PROCESSED), exist_ok=True)
    df.to_csv(config.DATA_PATH_PROCESSED, index=False)
    
    return df
