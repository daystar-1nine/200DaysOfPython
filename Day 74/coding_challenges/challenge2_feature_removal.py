"""
Day 74 - Challenge 2: Feature Removal Experiment
Compares all features vs removing a feature with overlapping information.
"""
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "advertising_sales.csv")).dropna()
    df = df[df["Sales"] > 0]
    
    y = df["Sales"]
    X_all = df[["TV_Spend", "Digital_Spend", "Radio_Spend", "Advertising_Spend", "Discount"]]
    X_sub = df[["TV_Spend", "Digital_Spend", "Radio_Spend", "Discount"]] # removed redundant Advertising_Spend
    
    X_tr1, X_te1, y_tr, y_te = train_test_split(X_all, y, test_size=0.2, random_state=42)
    X_tr2, X_te2, _, _ = train_test_split(X_sub, y, test_size=0.2, random_state=42)
    
    m1 = LinearRegression().fit(X_tr1, y_tr)
    m2 = LinearRegression().fit(X_tr2, y_tr)
    
    p1 = m1.predict(X_te1)
    p2 = m2.predict(X_te2)
    
    print(f"Model A (With Collinear Advertising_Spend): R^2 = {r2_score(y_te, p1):.4f}, RMSE = {root_mean_squared_error(y_te, p1):.2f}")
    print(f"Model B (Advertising_Spend Removed):        R^2 = {r2_score(y_te, p2):.4f}, RMSE = {root_mean_squared_error(y_te, p2):.2f}")
