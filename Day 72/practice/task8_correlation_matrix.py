"""
Day 72 — Task 8: E-Commerce Correlation Matrix
Computes the pairwise correlation matrix across 8 e-commerce variables and exports to CSV.
"""
import os
import pandas as pd

def main():
    data_path = os.path.join(r"s:\Programming\Python200days\Day 72\data", "ecommerce_sales.csv")
    out_dir = r"s:\Programming\Python200days\Day 72\output"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "correlation_matrix.csv")
    
    df = pd.read_csv(data_path)
    
    target_cols = [
        "Quantity", "Unit_Price", "Cost_Price", "Discount",
        "Revenue", "Cost", "Profit", "Profit_Margin"
    ]
    
    subset = df[target_cols]
    corr_matrix = subset.corr(method="pearson")
    corr_matrix.to_csv(out_csv)
    
    print("=" * 65)
    print("DAY 72 — TASK 8: E-COMMERCE CORRELATION MATRIX")
    print("=" * 65)
    print(corr_matrix.round(3).to_string())
    print(f"\nSaved correlation matrix to: {out_csv}")

if __name__ == "__main__":
    main()
