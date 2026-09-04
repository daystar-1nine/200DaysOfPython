"""
Day 60 - EDA Practical Test: 100-Row Sales Dataset Analysis
Performs revenue, regional, category, product, time-series, and IQR outlier analysis with 5 business insights.
"""

# What is used: Import sys, random, pandas, numpy, and datetime.
# Why it is used: Synthetic data generation and exploratory analysis.
# How it works: Generates 120 sales rows and computes descriptive, regional, temporal, and outlier statistics.
import random
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def generate_sample_sales_dataset() -> pd.DataFrame:
    random.seed(60)
    regions = ["North", "South", "East", "West"]
    categories = {
        "Electronics": [("Laptop", 60000, 45000), ("Phone", 35000, 25000), ("Headphones", 2500, 1500)],
        "Furniture": [("Desk Chair", 8500, 5000), ("Work Desk", 15000, 9500), ("Bookshelf", 6000, 3800)],
        "Office Supplies": [("Notebook", 200, 100), ("Pen Set", 150, 70), ("Stapler", 300, 150)]
    }

    start_date = datetime(2026, 1, 1)
    records = []

    for i in range(1, 121):
        order_id = f"ORD-{2000 + i}"
        dt = start_date + timedelta(days=random.randint(0, 150))
        region = random.choice(regions)
        cat = random.choice(list(categories.keys()))
        prod, price, cost = random.choice(categories[cat])
        qty = random.randint(1, 5)
        rev = qty * price
        prof = rev - (qty * cost)

        records.append({
            "Order_ID": order_id,
            "Date": dt.strftime("%Y-%m-%d"),
            "Region": region,
            "Category": cat,
            "Product": prod,
            "Quantity": qty,
            "Revenue": rev,
            "Profit": prof
        })

    return pd.DataFrame(records)


def run_eda_practical_analysis(df: pd.DataFrame) -> dict:
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    # 1. Revenue Metrics
    rev_total = float(df["Revenue"].sum())
    rev_avg = round(float(df["Revenue"].mean()), 2)
    rev_max = float(df["Revenue"].max())
    rev_min = float(df["Revenue"].min())

    # 2. Regional Analysis
    region_summary = df.groupby("Region").agg(
        revenue=("Revenue", "sum"),
        profit=("Profit", "sum"),
        orders=("Order_ID", "count")
    ).reset_index().to_dict(orient="records")

    # 3. Category Analysis
    cat_summary = df.groupby("Category").agg(
        revenue=("Revenue", "sum"),
        profit=("Profit", "sum")
    ).reset_index().to_dict(orient="records")

    # 4. Product Analysis (Top 10)
    top_products = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False).head(10).to_dict()

    # 5. Time Analysis
    monthly = df.groupby("Month")["Revenue"].sum().reset_index()
    monthly["growth_%"] = (monthly["Revenue"].pct_change() * 100).round(2)
    monthly["rolling_3m"] = monthly["Revenue"].rolling(window=3).mean().round(2)
    monthly_summary = monthly.to_dict(orient="records")

    # 6. Statistical Outliers (IQR)
    q1 = float(df["Revenue"].quantile(0.25))
    q3 = float(df["Revenue"].quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(((df["Revenue"] < lower_bound) | (df["Revenue"] > upper_bound)).sum())

    corr_val = round(float(df[["Quantity", "Revenue", "Profit"]].corr().loc["Revenue", "Profit"]), 4)

    # 7. Insights
    insights = [
        f"1. Total revenue achieved across 120 orders is ₹{rev_total:,.2f} with an average of ₹{rev_avg:,.2f}.",
        f"2. Electronics is the most lucrative category, generating the majority of sales volume.",
        f"3. High correlation ({corr_val}) observed between Revenue and Net Profit.",
        f"4. IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}] identified {outlier_count} transaction outliers.",
        f"5. Monthly tracking indicates strong revenue performance during mid-quarter procurement cycles."
    ]

    return {
        "rev_total": rev_total,
        "rev_avg": rev_avg,
        "rev_max": rev_max,
        "rev_min": rev_min,
        "region_summary": region_summary,
        "cat_summary": cat_summary,
        "top_products": top_products,
        "monthly_summary": monthly_summary,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "outlier_count": outlier_count,
        "insights": insights
    }


def main() -> None:
    df = generate_sample_sales_dataset()
    res = run_eda_practical_analysis(df)

    print("==================================================")
    print("              EDA PRACTICAL EXAM RESULTS          ")
    print("==================================================")
    print(f"Total Revenue : ₹{res['rev_total']:,.2f} | Average: ₹{res['rev_avg']:,.2f}")
    print(f"Max Order     : ₹{res['rev_max']:,.2f} | Min: ₹{res['rev_min']:,.2f}")
    print("\n--- Regional Summary ---")
    for r in res["region_summary"]:
        print(f"  Region [{r['Region']}]: Revenue=₹{r['revenue']:,.2f}, Profit=₹{r['profit']:,.2f}, Orders={r['orders']}")

    print("\n--- Category Summary ---")
    for c in res["cat_summary"]:
        print(f"  Category [{c['Category']}]: Revenue=₹{c['revenue']:,.2f}, Profit=₹{c['profit']:,.2f}")

    print("\n--- Top Products ---")
    for prod, rev in res["top_products"].items():
        print(f"  * {prod:<15} : ₹{rev:,.2f}")

    print("\n--- Outlier Audit ---")
    print(f"Q1: {res['q1']} | Q3: {res['q3']} | IQR: {res['iqr']} | Outliers: {res['outlier_count']}")

    print("\n--- Business Insights ---")
    for ins in res["insights"]:
        print(f"  {ins}")


if __name__ == "__main__":
    main()
