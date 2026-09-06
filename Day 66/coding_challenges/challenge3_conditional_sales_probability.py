"""
Day 66 Coding Challenge 3: Empirical Conditional Sales Probabilities
Analyzes real e-commerce transactional data to compute marginal, joint,
and conditional probabilities of high-profit sales given region and category,
validating Bayes' Rule on empirical data.
"""
from pathlib import Path
import pandas as pd

def compute_sales_conditional_probabilities(data_path: str | Path) -> dict:
    df = pd.read_csv(data_path)
    total_orders = len(df)
    
    # Define "High Profit" as profit above 75th percentile
    profit_threshold = df["Profit"].quantile(0.75)
    df["Is_High_Profit"] = df["Profit"] >= profit_threshold
    
    p_high_profit = df["Is_High_Profit"].mean()
    
    # Regional probabilities
    regional_metrics = {}
    for region, group in df.groupby("Region"):
        p_region = len(group) / total_orders
        p_high_given_region = group["Is_High_Profit"].mean()
        p_joint = (group["Is_High_Profit"].sum()) / total_orders
        # Bayes check: P(Region | High Profit) = P(High Profit | Region) * P(Region) / P(High Profit)
        p_region_given_high = (p_high_given_region * p_region) / p_high_profit if p_high_profit > 0 else 0
        
        regional_metrics[region] = {
            "P(Region)": round(p_region, 4),
            "P(HighProfit | Region)": round(p_high_given_region, 4),
            "P(HighProfit and Region)": round(p_joint, 4),
            "P(Region | HighProfit)": round(p_region_given_high, 4)
        }
        
    # Category probabilities
    category_metrics = {}
    for cat, group in df.groupby("Category"):
        category_metrics[cat] = {
            "P(Cat)": round(len(group) / total_orders, 4),
            "P(HighProfit | Cat)": round(group["Is_High_Profit"].mean(), 4)
        }
        
    return {
        "total_records": total_orders,
        "profit_threshold_p75": round(profit_threshold, 2),
        "P(HighProfit)": round(p_high_profit, 4),
        "regional_metrics": regional_metrics,
        "category_metrics": category_metrics
    }

def main():
    print("=" * 70)
    print("  CHALLENGE 3: CONDITIONAL & BAYESIAN SALES PROBABILITIES")
    print("=" * 70)
    
    csv_path = Path(__file__).resolve().parent.parent / "data" / "ecommerce_sales.csv"
    res = compute_sales_conditional_probabilities(csv_path)
    
    print(f"Total Transactions: {res['total_records']}")
    print(f"High Profit Cutoff (75th percentile): Rs. {res['profit_threshold_p75']}")
    print(f"Marginal P(High Profit): {res['P(HighProfit)'] * 100:.1f}%")
    
    print("\n[Regional Conditional Probabilities]")
    reg_df = pd.DataFrame(res["regional_metrics"]).T
    print(reg_df.to_string())
    
    print("\n[Category Conditional Probabilities]")
    cat_df = pd.DataFrame(res["category_metrics"]).T
    print(cat_df.to_string())

if __name__ == "__main__":
    main()
