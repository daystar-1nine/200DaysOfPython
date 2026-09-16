
import pandas as pd
import numpy as np

def clean_data(df):
    df_clean = df.copy()
    if 'CustomerID' in df_clean.columns:
        df_clean = df_clean.drop('CustomerID', axis=1)
        
    for col in df_clean.select_dtypes(include=['float64', 'int64']).columns:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    for col in df_clean.select_dtypes(include=['object']).columns:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            
    return df_clean
