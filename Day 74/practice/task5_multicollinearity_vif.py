"""
Day 74 - Practical Task 5: Multicollinearity & Variance Inflation Factor (VIF)
Computes correlation matrix and VIF scores for advertising channels.
"""
import os
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "advertising_sales.csv")
    df = pd.read_csv(data_path).dropna()
    
    num_cols = ["TV_Spend", "Digital_Spend", "Radio_Spend", "Discount", "Quantity"]
    corr = df[num_cols].corr()
    print("=== Correlation Matrix ===")
    print(corr.round(3))
    
    X = add_constant(df[num_cols].dropna())
    vifs = [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]
    vif_df = pd.DataFrame({"Feature": num_cols, "VIF": np.round(vifs, 2)})
    print("\n=== VIF Scores ===")
    print(vif_df)
