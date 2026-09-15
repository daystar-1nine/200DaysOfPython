import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df["Charges_Per_Month_Ratio"] = df["Total_Charges"] / (df["Tenure_Months"] * df["Monthly_Charges"] + 1e-6)
    df["Support_Per_Tenure"] = df["Support_Calls"] / (df["Tenure_Months"] + 1)
    df["Complaint_Risk_Index"] = (
        df["Complaints"] * 2.0 +
        df["Late_Payments"] * 1.5 +
        df["Support_Calls"] * 1.0
    )
    df["Net_Monthly_Charges"] = df["Monthly_Charges"] * (1.0 - (df["Discount"] / 100.0))
    df["Tenure_Age_Ratio"] = df["Tenure_Months"] / (df["Age"] + 1e-6)
    
    return df
