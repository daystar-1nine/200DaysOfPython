from typing import Tuple, List
import pandas as pd

REQUIRED_COLUMNS = [
    'Customer_ID', 'Age', 'Gender', 'Tenure_Months', 'Monthly_Charges',
    'Total_Charges', 'Contract_Type', 'Internet_Service', 'Payment_Method',
    'Support_Calls', 'Late_Payments', 'Usage_Hours', 'Discount', 'Complaints', 'Churn'
]

def validate_raw_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors = []
    if df is None or df.empty:
        return False, ["DataFrame is null or empty"]
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    if 'Churn' in df.columns:
        unique_vals = set(df['Churn'].dropna().unique())
        if not unique_vals.issubset({0, 1, 0.0, 1.0}):
            errors.append(f"Target column 'Churn' has invalid non-binary values: {unique_vals}")
    return len(errors) == 0, errors

def validate_cleaned_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors = []
    if df is None or df.empty:
        return False, ["Cleaned DataFrame is null or empty"]
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        bad_cols = null_counts[null_counts > 0].to_dict()
        errors.append(f"Cleaned DataFrame contains residual nulls: {bad_cols}")
    if 'Churn' in df.columns:
        if not set(df['Churn'].unique()).issubset({0, 1}):
            errors.append("Cleaned Churn column contains values other than 0 and 1")
    return len(errors) == 0, errors
