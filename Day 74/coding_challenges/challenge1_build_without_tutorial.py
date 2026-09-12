"""
Day 74 - Challenge 1: Build Multiple Regression Without Tutorial
Build an end-to-end model with TV, Digital, Radio, Discount, and Region.
"""
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "advertising_sales.csv")).dropna()
    df = df[df["Sales"] > 0]
    df = df[df["Region"] != "Unknown"]
    
    num_features = ["TV_Spend", "Digital_Spend", "Radio_Spend", "Discount"]
    cat_features = ["Region"]
    
    X = df[num_features + cat_features]
    y = df["Sales"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipe = Pipeline([
        ("pre", ColumnTransformer([
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_features)
        ])),
        ("reg", LinearRegression())
    ])
    
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print(f"Challenge 1 Pipeline R^2: {r2_score(y_test, preds):.4f}")
    print(f"Challenge 1 Pipeline RMSE: {root_mean_squared_error(y_test, preds):.2f}")
