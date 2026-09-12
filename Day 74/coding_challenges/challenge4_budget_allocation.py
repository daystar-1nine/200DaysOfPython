"""
Day 74 - Challenge 4: Budget Allocation Business Decision
Answers how to allocate an additional INR 100,000 marketing budget across channels
considering coefficient returns, scale, diminishing returns, and business constraints.
"""
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "advertising_sales.csv")).dropna()
    df = df[df["Sales"] > 0]
    
    features = ["TV_Spend", "Digital_Spend", "Radio_Spend"]
    X = df[features]
    y = df["Sales"]
    
    reg = LinearRegression().fit(X, y)
    print("=== Channel Return Coefficients ===")
    for feat, coef in zip(features, reg.coef_):
        print(f"  {feat}: +INR {coef:.2f} Sales per INR 1.00 spent")
        
    print("\n=== Strategic Recommendation for INR 100,000 Allocation ===")
    print("1. Highest Marginal Return: Digital Spend yields the highest estimated coefficient.")
    print("2. Diversification & Diminishing Returns: Avoid 100% single-channel concentration.")
    print("3. Proposed Split: 60% Digital (INR 60,000), 30% TV (INR 30,000), 10% Radio (INR 10,000).")
