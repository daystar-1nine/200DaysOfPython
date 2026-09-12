import pandas as pd
import numpy as np
from app.cleaner import clean_data

def test_clean_standardizes_contract():
    df = pd.DataFrame({
        "Customer_ID": ["C1", "C2"],
        "Contract_Type": ["month-to-month", "Month to month"]
    })
    res = clean_data(df)
    assert (res["Contract_Type"] == "Month-to-month").all()

def test_clean_removes_negative_values():
    df = pd.DataFrame({
        "Customer_ID": ["C1", "C2"],
        "Age": [-5, 30],
        "Monthly_Charges": [50.0, -10.0]
    })
    res = clean_data(df)
    assert len(res) == 0

def test_clean_imputes_missing():
    df = pd.DataFrame({
        "Customer_ID": ["C1", "C2", "C3"],
        "Age": [20.0, np.nan, 40.0],
        "Monthly_Charges": [50.0, 60.0, np.nan],
        "Tenure_Months": [10, 10, 10],
        "Total_Charges": [500.0, np.nan, 600.0]
    })
    res = clean_data(df)
    assert res["Age"].isna().sum() == 0
    assert res["Monthly_Charges"].isna().sum() == 0
    assert res["Total_Charges"].isna().sum() == 0

def test_clean_drops_duplicates():
    df = pd.DataFrame({
        "Customer_ID": ["C1", "C1", "C2"],
        "Age": [25, 25, 30]
    })
    res = clean_data(df)
    assert len(res) == 2

def test_clean_valid_targets():
    df = pd.DataFrame({
        "Customer_ID": ["C1", "C2", "C3"],
        "Churn": [0, 1, 99],
        "Risk_Level": ["low", "HIGH", "invalid"]
    })
    res = clean_data(df)
    assert set(res["Churn"].unique()).issubset({0, 1})
    assert set(res["Risk_Level"].unique()).issubset({"Low", "Medium", "High"})
