"""
Day 59 - Coding Challenge 3: Time-Series Growth & Moving Average Analytics
Computes previous period values, absolute differences, percentage growth rates, and 3-period rolling averages.
"""

# What is used: Import sys and pandas modules.
# Why it is used: Configures UTF-8 console output and performs time-series analytics.
# How it works: Brings sys and pandas namespaces into scope.
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def compute_time_series_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time-series lag, growth rate, and rolling average metrics.

    Args:
        df: Input DataFrame containing 'Period' and 'Sales' columns.

    Returns:
        pd.DataFrame: Enriched DataFrame with Prev_Sales, Sales_Diff, Growth_Pct, and Rolling_3_Avg.
    """
    res = df.copy()

    # What is used: shift(1), diff(), pct_change(), and rolling(3).mean().
    # Why it is used: Computes lag value, difference, percentage growth, and 3-period moving average.
    # How it works: Applies time-series operations down rows in sequential order.
    res["Prev_Sales"] = res["Sales"].shift(1)
    res["Sales_Diff"] = res["Sales"].diff().round(2)
    res["Growth_Pct"] = (res["Sales"].pct_change() * 100.0).round(2)
    res["Rolling_3_Avg"] = res["Sales"].rolling(window=3).mean().round(2)

    return res


def main() -> None:
    data = {
        "Period": ["2026-Q1", "2026-Q2", "2026-Q3", "2026-Q4", "2027-Q1", "2027-Q2"],
        "Sales": [150000, 180000, 175000, 210000, 240000, 230000]
    }
    df = pd.DataFrame(data)
    analyzed = compute_time_series_metrics(df)
    print("--- Quarterly Sales Growth & 3-Period Moving Average ---")
    print(analyzed)


if __name__ == "__main__":
    main()
