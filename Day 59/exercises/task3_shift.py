"""
Day 59 - Exercise 3: Time-Series Lag & Month-over-Month Growth with shift()
Demonstrates lag comparisons, absolute differences, and percentage growth calculations.
"""

# What is used: Import sys and pandas modules.
# Why it is used: Configures UTF-8 console output and computes lag differences.
# How it works: Brings sys and pandas namespaces into scope.
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    # What is used: Sample monthly revenue DataFrame.
    # Why it is used: Provides chronological sequence of sales records.
    # How it works: Creates DataFrame indexed by Month with Revenue figures.
    data = {
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "Revenue": [100000, 120000, 115000, 140000, 160000, 155000]
    }
    df = pd.DataFrame(data)

    # What is used: df["Revenue"].shift(1).
    # Why it is used: Exposes previous month's revenue in current row.
    # How it works: Shifts values down by 1 row, populating index 0 with NaN.
    df["Previous_Month_Revenue"] = df["Revenue"].shift(1)

    # What is used: Subtraction and pct_change().
    # Why it is used: Calculates absolute revenue change and percentage growth rate.
    # How it works: Subtracts Previous_Month_Revenue from Revenue and calculates percentage change.
    df["Revenue_Change"] = df["Revenue"] - df["Previous_Month_Revenue"]
    df["Growth_Percentage"] = (df["Revenue"].pct_change() * 100.0).round(2)

    print("--- Month-over-Month Revenue Growth Analysis ---")
    print(df)


if __name__ == "__main__":
    main()
