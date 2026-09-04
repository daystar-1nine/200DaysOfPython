"""
Day 59 - Exercise 4: 7-Day Rolling Average Trend Analysis
Demonstrates moving window analytics using rolling(window=7).mean().
"""

# What is used: Import sys, pandas, and numpy modules.
# Why it is used: Configures UTF-8 console output and generates rolling statistics.
# How it works: Brings sys, pandas, and numpy namespaces into scope.
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    # What is used: pd.date_range and np.random.randint.
    # Why it is used: Generates 30 days of synthetic daily sales data.
    # How it works: Creates daily dates from 2026-01-01 to 2026-01-30 with random sales values.
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=30, freq="D")
    sales = np.random.randint(1000, 5000, size=30)
    df = pd.DataFrame({"Date": dates, "Daily_Sales": sales})

    # What is used: df["Daily_Sales"].rolling(window=7).mean().
    # Why it is used: Smoothes daily fluctuations using a 7-day moving average.
    # How it works: Calculates mean of current row and preceding 6 rows.
    df["7_Day_Rolling_Avg"] = df["Daily_Sales"].rolling(window=7).mean().round(2)

    print("--- 30-Day Daily Sales with 7-Day Rolling Average (Top 15 Rows) ---")
    print(df.head(15))


if __name__ == "__main__":
    main()
