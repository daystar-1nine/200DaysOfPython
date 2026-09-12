import pytest
import numpy as np
import pandas as pd
from app.config import AppConfig

@pytest.fixture
def config():
    return AppConfig()

@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 200
    cids = [f"CUST_{i:04d}" for i in range(n)]
    ages = np.random.randint(20, 70, size=n)
    genders = np.random.choice(["Male", "Female"], size=n)
    contracts = np.random.choice(["Month-to-month", "One year", "Two year"], size=n)
    internet = np.random.choice(["Fiber optic", "DSL", "No"], size=n)
    payments = np.random.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], size=n)
    tenure = np.random.randint(1, 60, size=n)
    monthly = np.random.uniform(20.0, 100.0, size=n).round(2)
    total = (tenure * monthly).round(2)
    calls = np.random.randint(0, 6, size=n)
    late = np.random.randint(0, 4, size=n)
    usage = np.random.uniform(50.0, 250.0, size=n).round(1)
    discount = np.random.choice([0.0, 10.0, 20.0], size=n)
    complaints = np.random.randint(0, 3, size=n)
    churn = np.random.choice([0, 1], size=n, p=[0.75, 0.25])
    risk = np.where(churn == 1, np.random.choice(["Medium", "High"], size=n), np.random.choice(["Low", "Medium"], size=n))
    
    return pd.DataFrame({
        "Customer_ID": cids,
        "Age": ages,
        "Gender": genders,
        "Tenure_Months": tenure,
        "Monthly_Charges": monthly,
        "Total_Charges": total,
        "Contract_Type": contracts,
        "Internet_Service": internet,
        "Payment_Method": payments,
        "Support_Calls": calls,
        "Late_Payments": late,
        "Usage_Hours": usage,
        "Discount": discount,
        "Complaints": complaints,
        "Risk_Level": risk,
        "Churn": churn
    })
