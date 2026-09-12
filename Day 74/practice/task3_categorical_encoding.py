"""
Day 74 - Practical Task 3: Categorical Variables & One-Hot Encoding
Demonstrates handling numeric and categorical predictors using ColumnTransformer.
"""
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "advertising_sales.csv")
    df = pd.read_csv(data_path).dropna()
    df = df[df["Sales"] > 0]
    df = df[df["Region"] != "Unknown"]
    
    num_cols = ["TV_Spend", "Digital_Spend", "Radio_Spend", "Discount"]
    cat_cols = ["Region", "Category"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ("prep", preprocessor),
        ("reg", LinearRegression())
    ])
    
    X = df[num_cols + cat_cols]
    y = df["Sales"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    
    r2 = pipeline.score(X_test, y_test)
    print(f"Pipeline R^2 with Categorical Encoding: {r2:.4f}")
