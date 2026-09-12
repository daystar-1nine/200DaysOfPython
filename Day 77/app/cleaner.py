import pandas as pd
from app.config import AppConfig

def clean_data(df: pd.DataFrame, config: AppConfig = None) -> pd.DataFrame:
    if config is None:
        config = AppConfig()
        
    df = df.copy()
    
    # 1. Standardize Contract_Type
    if "Contract_Type" in df.columns:
        df["Contract_Type"] = df["Contract_Type"].replace({
            "month-to-month": "Month-to-month",
            "Month to month": "Month-to-month",
            "one year": "One year",
            "two year": "Two year"
        })
        
    # 2. Filter out invalid negative values
    if "Age" in df.columns:
        df = df[df["Age"] >= 0]
    if "Monthly_Charges" in df.columns:
        df = df[df["Monthly_Charges"] >= 0]
    if "Tenure_Months" in df.columns:
        df = df[df["Tenure_Months"] >= 0]
        
    # 3. Impute missing values without chained assignment
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Monthly_Charges" in df.columns:
        df["Monthly_Charges"] = df["Monthly_Charges"].fillna(df["Monthly_Charges"].median())
    if "Total_Charges" in df.columns and "Tenure_Months" in df.columns and "Monthly_Charges" in df.columns:
        df["Total_Charges"] = df["Total_Charges"].fillna(df["Tenure_Months"] * df["Monthly_Charges"])
        
    # 4. Drop duplicates by Customer_ID
    if "Customer_ID" in df.columns:
        df = df.drop_duplicates(subset=["Customer_ID"])
        
    # 5. Clean Churn binary targets
    if "Churn" in df.columns:
        df = df.dropna(subset=["Churn"])
        df = df[df["Churn"].isin([0, 1])]
        df["Churn"] = df["Churn"].astype(int)
        
    # 6. Standardize Risk_Level targets
    if "Risk_Level" in df.columns:
        df = df.dropna(subset=["Risk_Level"])
        df["Risk_Level"] = df["Risk_Level"].astype(str).str.capitalize()
        df = df[df["Risk_Level"].isin(["Low", "Medium", "High"])]
        
    df.to_csv(config.DATA_PATH_PROCESSED, index=False)
    return df
