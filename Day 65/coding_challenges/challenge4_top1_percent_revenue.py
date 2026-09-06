"""
Challenge 4 — Top 1% Revenue Transaction Dissection
Isolates top 1% transactions by Revenue and compares financial profiles with the remaining 99%.
"""

import os
import pandas as pd
import numpy as np

csv_path = r"s:\Programming\Python200days\Day 65\data\ecommerce_sales.csv"
if not os.path.exists(csv_path):
    csv_path = r"s:\Programming\Python200days\Day 64\data\ecommerce_sales.csv"

df = pd.read_csv(csv_path)

# 99th percentile threshold
p99_threshold = df["Revenue"].quantile(0.99)
top_1pct = df[df["Revenue"] >= p99_threshold]
remaining_99pct = df[df["Revenue"] < p99_threshold]

summary = {
    "Segment": ["Top 1% VIP Orders", "Remaining 99% Orders", "Ratio (Top 1% / Remaining)"],
    "Order_Count": [len(top_1pct), len(remaining_99pct), f"{(len(top_1pct) / len(df)) * 100:.1f}% of total"],
    "Avg_Revenue": [round(top_1pct["Revenue"].mean(), 2), round(remaining_99pct["Revenue"].mean(), 2), round(top_1pct["Revenue"].mean() / remaining_99pct["Revenue"].mean(), 2)],
    "Avg_Profit": [round(top_1pct["Profit"].mean(), 2), round(remaining_99pct["Profit"].mean(), 2), round(top_1pct["Profit"].mean() / remaining_99pct["Profit"].mean(), 2)],
    "Avg_Quantity": [round(top_1pct["Quantity"].mean(), 2), round(remaining_99pct["Quantity"].mean(), 2), round(top_1pct["Quantity"].mean() / remaining_99pct["Quantity"].mean(), 2)],
    "Avg_Discount": [round(top_1pct["Discount"].mean(), 3), round(remaining_99pct["Discount"].mean(), 3), round(top_1pct["Discount"].mean() / remaining_99pct["Discount"].mean(), 2)],
}

summary_df = pd.DataFrame(summary)

print("=== CHALLENGE 4: TOP 1% REVENUE ANALYSIS ===")
print(f"P99 Revenue Threshold: Rs. {p99_threshold:,.2f}")
print(summary_df.to_string(index=False))
print("\nKey Takeaway:")
print(f"Top 1% orders generate {summary_df.loc[2, 'Avg_Revenue']}x more revenue and {summary_df.loc[2, 'Avg_Profit']}x more profit per transaction.")
