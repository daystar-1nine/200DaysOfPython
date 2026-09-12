import pandas as pd
from app.config import AppConfig

def validate_data(df: pd.DataFrame, config: AppConfig = None, min_rows: int = None) -> bool:
    if df.empty:
        raise ValueError("DataFrame is empty")
        
    if min_rows is not None and len(df) < min_rows:
        raise ValueError(f"DataFrame has fewer than {min_rows} rows")
        
    if "Age" in df.columns and (df["Age"] < 0).any():
        raise ValueError("Negative Age values found")
        
    if "Monthly_Charges" in df.columns and (df["Monthly_Charges"] < 0).any():
        raise ValueError("Negative Monthly_Charges values found")
        
    if "Tenure_Months" in df.columns and (df["Tenure_Months"] < 0).any():
        raise ValueError("Negative Tenure_Months values found")
        
    if "Churn" in df.columns:
        invalid_churn = df[~df["Churn"].isin([0, 1])]
        if not invalid_churn.empty:
            raise ValueError("Invalid Churn values found")
            
    return True
