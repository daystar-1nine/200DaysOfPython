"""
Day 74 - Practical Task 2: Multiple Regression Demo
End-to-end multiple regression workflow using numerical features:
TV_Spend, Digital_Spend, Radio_Spend, Discount -> Sales
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "advertising_sales.csv")
    df = pd.read_csv(data_path).dropna()
    df = df[df["Sales"] > 0]
    
    features = ["TV_Spend", "Digital_Spend", "Radio_Spend", "Discount"]
    X = df[features]
    y = df["Sales"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    print("=== Model Coefficients ===")
    print(f"Intercept: {model.intercept_:.2f}")
    for feat, coef in zip(features, model.coef_):
        print(f"  {feat}: {coef:.4f}")
        
    print("\n=== Test Set Metrics ===")
    print(f"MAE:  {mean_absolute_error(y_test, preds):.2f}")
    print(f"RMSE: {root_mean_squared_error(y_test, preds):.2f}")
    print(f"R^2:  {r2_score(y_test, preds):.4f}")
