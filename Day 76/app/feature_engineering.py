import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    if "Usage_Hours" in df.columns and "Tenure_Months" in df.columns:
        df["Avg_Monthly_Usage"] = df["Usage_Hours"] / (df["Tenure_Months"].clip(lower=1))
        
    if "Monthly_Charges" in df.columns and "Support_Calls" in df.columns:
        df["Charges_Per_Call"] = df["Monthly_Charges"] / (df["Support_Calls"] + 1)
        
    if "Support_Calls" in df.columns and "Late_Payments" in df.columns:
        df["Support_Risk_Index"] = df["Support_Calls"] * df["Late_Payments"]
        
    return df
