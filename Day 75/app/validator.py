import pandas as pd
from typing import Optional
from app.config import AppConfig

def validate_data(df: pd.DataFrame, config: Optional[AppConfig] = None, min_rows: Optional[int] = None) -> bool:
    if config is None:
        config = AppConfig()
        
    if min_rows and len(df) < min_rows:
        raise ValueError(f"Data has {len(df)} rows, minimum required is {min_rows}")
        
    if (df[config.TARGET_COL] < 0).any():
        raise ValueError("Negative sales found")
        
    if (df['Discount'] < 0).any() or (df['Discount'] > 100).any():
        raise ValueError("Invalid discount found")
        
    return True
