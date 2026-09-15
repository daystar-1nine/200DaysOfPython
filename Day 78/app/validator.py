import pandas as pd
from app.config import AppConfig

def validate_data(df: pd.DataFrame, config: AppConfig = None) -> bool:
    if config is None:
        config = AppConfig()
        
    if df is None or df.empty:
        raise ValueError("Dataset is None or empty.")
        
    required_cols = [
        "Customer_ID", "Age", "Gender", "Tenure_Months", "Monthly_Charges",
        "Total_Charges", "Contract_Type", "Internet_Service", "Payment_Method",
        "Support_Calls", "Late_Payments", "Usage_Hours", "Discount", "Complaints",
        "Churn"
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
        
    if (df["Age"] < 0).any():
        raise ValueError("Negative values found in Age.")
    if (df["Monthly_Charges"] < 0).any():
        raise ValueError("Negative values found in Monthly_Charges.")
    if (df["Tenure_Months"] < 0).any():
        raise ValueError("Negative values found in Tenure_Months.")
        
    invalid_genders = set(df["Gender"].unique()) - set(config.VALID_GENDERS)
    if invalid_genders:
        raise ValueError(f"Invalid Gender values: {invalid_genders}")
        
    invalid_contracts = set(df["Contract_Type"].unique()) - set(config.VALID_CONTRACTS)
    if invalid_contracts:
        raise ValueError(f"Invalid Contract_Type values: {invalid_contracts}")
        
    invalid_churn = set(df["Churn"].unique()) - {0, 1}
    if invalid_churn:
        raise ValueError(f"Invalid Churn values: {invalid_churn}")
        
    return True
