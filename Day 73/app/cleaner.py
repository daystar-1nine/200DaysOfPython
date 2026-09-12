"""
Day 73 - Dataset Cleaner
Cleans raw sales data by filtering nulls, negative values, and duplicate records.
"""
import pandas as pd

def clean_sales_data(df: pd.DataFrame, feature_col: str = "Advertising_Spend", target_col: str = "Sales") -> pd.DataFrame:
    cleaned = df.copy()
    
    # 1. Drop duplicates
    cleaned = cleaned.drop_duplicates()
    
    # 2. Drop rows with nulls in key columns
    cleaned = cleaned.dropna(subset=[feature_col, target_col])
    
    # 3. Filter out non-positive spend and sales
    cleaned = cleaned[(cleaned[feature_col] > 0) & (cleaned[target_col] > 0)]
    
    if cleaned.empty:
        raise ValueError("Dataset has no valid positive records after cleaning.")
        
    return cleaned.reset_index(drop=True)
