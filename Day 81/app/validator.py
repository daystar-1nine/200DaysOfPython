
import pandas as pd

def validate_data(df):
    if df.empty:
        raise ValueError("Dataset is empty.")
    if 'Churn' not in df.columns:
        raise ValueError("Target column 'Churn' not found.")
    return True
