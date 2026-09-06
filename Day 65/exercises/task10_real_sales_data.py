"""
Task 10 — Descriptive Statistics Summary Table on Real Sales Data
Analyze Revenue, Profit, Quantity, Unit_Price, and Discount from ecommerce_sales.csv.
"""

import os
import pandas as pd
import numpy as np

csv_path = r"s:\Programming\Python200days\Day 65\data\ecommerce_sales.csv"
if not os.path.exists(csv_path):
    # Fallback to Day 64 if not yet copied
    csv_path = r"s:\Programming\Python200days\Day 64\data\ecommerce_sales.csv"

df = pd.read_csv(csv_path)

columns_to_analyze = ["Revenue", "Profit", "Quantity", "Unit_Price", "Discount"]
summary_rows = []

for col in columns_to_analyze:
    s = df[col].dropna()
    q1 = float(s.quantile(0.25))
    q2 = float(s.quantile(0.50))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    
    row = {
        "Column": col,
        "Count": int(s.count()),
        "Mean": round(float(s.mean()), 2),
        "Median": round(q2, 2),
        "Std": round(float(s.std()), 2),
        "Min": round(float(s.min()), 2),
        "Q1": round(q1, 2),
        "Q2": round(q2, 2),
        "Q3": round(q3, 2),
        "Max": round(float(s.max()), 2),
        "IQR": round(iqr, 2),
        "Skewness": round(float(s.skew()), 2),
        "Kurtosis": round(float(s.kurt()), 2),
    }
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

print("=== TASK 10: REAL SALES DATA STATISTICAL SUMMARY ===")
print(summary_df.to_string(index=False))
