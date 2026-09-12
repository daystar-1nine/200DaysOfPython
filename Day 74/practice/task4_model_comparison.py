"""
Day 74 - Practical Task 4: Model Comparison & Adjusted R^2
Compares 4 progressive models:
M1: TV
M2: TV + Digital
M3: TV + Digital + Radio
M4: TV + Digital + Radio + Discount + Region
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def adj_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "advertising_sales.csv")
    df = pd.read_csv(data_path).dropna()
    df = df[df["Sales"] > 0]
    df = df[df["Region"] != "Unknown"]
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    y_train = train_df["Sales"]
    y_test = test_df["Sales"]
    n = len(y_test)
    
    models = {
        "M1 (TV)": ["TV_Spend"],
        "M2 (TV + Digital)": ["TV_Spend", "Digital_Spend"],
        "M3 (+ Radio)": ["TV_Spend", "Digital_Spend", "Radio_Spend"],
    }
    
    results = []
    for name, feats in models.items():
        reg = LinearRegression().fit(train_df[feats], y_train)
        preds = reg.predict(test_df[feats])
        r2 = r2_score(y_test, preds)
        p = len(feats)
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": root_mean_squared_error(y_test, preds),
            "R2": r2,
            "Adj_R2": adj_r2(r2, n, p)
        })
        
    # M4 with Region
    m4_prep = ColumnTransformer(
        transformers=[
            ("num", "passthrough", ["TV_Spend", "Digital_Spend", "Radio_Spend", "Discount"]),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), ["Region"])
        ]
    )
    m4_pipe = Pipeline([("prep", m4_prep), ("reg", LinearRegression())])
    m4_pipe.fit(train_df[["TV_Spend", "Digital_Spend", "Radio_Spend", "Discount", "Region"]], y_train)
    m4_preds = m4_pipe.predict(test_df[["TV_Spend", "Digital_Spend", "Radio_Spend", "Discount", "Region"]])
    m4_r2 = r2_score(y_test, m4_preds)
    p4 = 4 + 3 # 4 numeric + 3 region dummy cols
    results.append({
        "Model": "M4 (+ Disc + Reg)",
        "MAE": mean_absolute_error(y_test, m4_preds),
        "RMSE": root_mean_squared_error(y_test, m4_preds),
        "R2": m4_r2,
        "Adj_R2": adj_r2(m4_r2, n, p4)
    })
    
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
