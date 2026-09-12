import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers new features for the dataset.
    """
    df = df.copy()
    
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        if "Month" not in df.columns:
            df["Month"] = df["Date"].dt.month
        if "Year" not in df.columns:
            df["Year"] = df["Date"].dt.year
            
    # Add Total_Ad_Spend
    ad_cols = ["TV_Spend", "Digital_Spend", "Radio_Spend"]
    existing_ad_cols = [c for c in ad_cols if c in df.columns]
    df["Total_Ad_Spend"] = df[existing_ad_cols].sum(axis=1)
    
    if "Advertising_Spend" not in df.columns:
        df["Advertising_Spend"] = df["Total_Ad_Spend"]
        
    # Add Ad_Efficiency
    if "Sales" in df.columns and "Advertising_Spend" in df.columns:
        df["Ad_Efficiency"] = np.where(df["Advertising_Spend"] > 0, df["Sales"] / df["Advertising_Spend"], 0)
        
    # Add Discount_Intensity
    if "Discount" in df.columns and "Sales" in df.columns:
        df["Discount_Intensity"] = (df["Discount"] / 100.0) * df["Sales"]
        
    return df
