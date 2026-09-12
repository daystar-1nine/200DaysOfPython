import pandas as pd
from app.config import AppConfig
import numpy as np

def clean_data(df: pd.DataFrame, config: AppConfig = None) -> pd.DataFrame:
    if config is None:
        config = AppConfig()
        
    df = df.copy()
    
    # Standardize Contract_Type
    if "Contract_Type" in df.columns:
        df["Contract_Type"] = df["Contract_Type"].replace(
            {"month-to-month": "Month-to-month", 
             "Month to month": "Month-to-month", 
             "month to month": "Month-to-month"}
        )
        
    # Drop negative values
    if "Age" in df.columns:
        df = df[df["Age"] >= 0]
    if "Monthly_Charges" in df.columns:
        df = df[df["Monthly_Charges"] >= 0]
    if "Tenure_Months" in df.columns:
        df = df[df["Tenure_Months"] >= 0]
        
    # Impute missing values
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Monthly_Charges" in df.columns:
        df["Monthly_Charges"] = df["Monthly_Charges"].fillna(df["Monthly_Charges"].median())
    if "Total_Charges" in df.columns and "Tenure_Months" in df.columns and "Monthly_Charges" in df.columns:
        df["Total_Charges"] = df["Total_Charges"].fillna(df["Tenure_Months"] * df["Monthly_Charges"])
        
    # Drop duplicates
    if "Customer_ID" in df.columns:
        df = df.drop_duplicates(subset=["Customer_ID"])
        
    # Drop rows where Churn is null or not in [0, 1]
    if "Churn" in df.columns:
        df = df.dropna(subset=["Churn"])
        df = df[df["Churn"].isin([0, 1])]
        df["Churn"] = df["Churn"].astype(int)
        
    df.to_csv(config.DATA_PATH_PROCESSED, index=False)
    return df
