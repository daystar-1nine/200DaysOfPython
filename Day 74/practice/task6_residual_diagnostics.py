"""
Day 74 - Practical Task 6: Residual Diagnostics
Evaluates residual zero-mean condition, variance, and distribution.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import skew, normaltest

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "advertising_sales.csv")
    df = pd.read_csv(data_path).dropna()
    df = df[df["Sales"] > 0]
    
    feats = ["TV_Spend", "Digital_Spend", "Radio_Spend"]
    X = df[feats]
    y = df["Sales"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = LinearRegression().fit(X_train, y_train)
    preds = reg.predict(X_test)
    residuals = y_test - preds
    
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    skew_res = skew(residuals)
    stat, p_val = normaltest(residuals)
    
    print("=== Residual Diagnostics ===")
    print(f"Mean of Residuals:   {mean_res:.4f} (Expect ~0)")
    print(f"Std of Residuals:    {std_res:.2f}")
    print(f"Skewness:            {skew_res:.4f}")
    print(f"Normality Test p-val:{p_val:.4f}")
