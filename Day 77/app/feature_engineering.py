import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Ratio of total charges to expected baseline charges
    df["Charges_Per_Month_Ratio"] = df["Total_Charges"] / (df["Tenure_Months"] * df["Monthly_Charges"] + 1e-6)
    
    # Support velocity relative to customer tenure
    df["Support_Per_Tenure"] = df["Support_Calls"] / (df["Tenure_Months"] + 1)
    
    # Comprehensive dissatisfaction index combining complaints, late payments, and support calls
    df["Complaint_Risk_Index"] = (
        df["Complaints"] * 2.0 +
        df["Late_Payments"] * 1.5 +
        df["Support_Calls"] * 1.0
    )
    
    # Net monthly charge after discounting
    df["Net_Monthly_Charges"] = df["Monthly_Charges"] * (1.0 - (df["Discount"] / 100.0))
    
    return df
