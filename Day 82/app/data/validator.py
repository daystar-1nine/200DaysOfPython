
import pandas as pd
def validate_data(df):
    if df.empty: raise ValueError("Empty dataset")
    if 'Churn' not in df.columns: raise ValueError("Target missing")
    return True
