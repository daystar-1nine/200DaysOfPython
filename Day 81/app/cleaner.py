
import pandas as pd

def clean_data(df):
    df_clean = df.copy()
    if 'CustomerID' in df_clean.columns:
        df_clean = df_clean.drop('CustomerID', axis=1)
    
    # Fill missing numeric values with median
    num_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    # Fill missing categorical values with mode
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            
    return df_clean
