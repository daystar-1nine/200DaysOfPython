import pandas as pd
import numpy as np

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean customer churn dataset: handle duplicates and missing values."""
    df_clean = df.copy()
    
    # Drop full duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Drop duplicates by Customer_ID if present
    if 'Customer_ID' in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=['Customer_ID'])
        
    # Impute numeric columns with median
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            
    # Impute categorical columns with mode
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            
    # Ensure Churn is integer binary (0 or 1)
    if 'Churn' in df_clean.columns:
        df_clean['Churn'] = df_clean['Churn'].astype(int)
        
    return df_clean
