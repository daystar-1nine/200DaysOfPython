import pandas as pd
import numpy as np
from typing import Dict, Any

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset: remove duplicates and impute missing values."""
    df_clean = df.copy()
    
    # Drop duplicates
    df_clean = df_clean.drop_duplicates()
    if 'Customer_ID' in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=['Customer_ID'])
        
    # Impute numeric columns with median
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            
    # Impute categorical columns with mode
    cat_cols = df_clean.select_dtypes(include=['object', 'category', 'string']).columns
    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            
    # Ensure Churn is int
    if 'Churn' in df_clean.columns:
        df_clean['Churn'] = df_clean['Churn'].astype(int)
        
    return df_clean

def generate_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate structured summary of dataset quality."""
    report = []
    total_rows = len(df)
    
    for col in df.columns:
        null_cnt = int(df[col].isnull().sum())
        null_pct = (null_cnt / total_rows) * 100.0 if total_rows > 0 else 0.0
        unique_cnt = int(df[col].nunique())
        dtype_str = str(df[col].dtype)
        
        report.append({
            'Column': col,
            'Data_Type': dtype_str,
            'Missing_Count': null_cnt,
            'Missing_Percentage': round(null_pct, 2),
            'Unique_Values': unique_cnt,
            'Sample_Value': str(df[col].iloc[0]) if total_rows > 0 else 'N/A'
        })
        
    return pd.DataFrame(report)
