"""
Day 57 - Coding Challenge 5: Regional Revenue Percentage Contribution
Calculate the percentage contribution of each region to the overall grand total revenue.
"""

# What is used: Import pandas library.
# Why it is used: Core package for Series vector arithmetic and percentage calculation.
# How it works: Brings pandas namespace into module scope.
import pandas as pd


def calculate_regional_revenue_percentage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total revenue per region and its percentage contribution to overall revenue.

    Formula: (Region_Revenue / Total_Revenue) * 100

    Args:
        df: Input DataFrame containing Region and Revenue columns.

    Returns:
        pd.DataFrame: DataFrame with Region, Total_Revenue, and Percentage_Contribution.
    """
    # What is used: df.groupby("Region")["Revenue"].sum().reset_index().
    # Why it is used: Sums total revenue per region.
    # How it works: Groups data by Region and sums Revenue Series.
    regional_summary = df.groupby("Region")["Revenue"].sum().reset_index()
    regional_summary.rename(columns={"Revenue": "Region_Revenue"}, inplace=True)

    # What is used: regional_summary["Region_Revenue"].sum().
    # Why it is used: Obtains total grand revenue scalar across all regions.
    # How it works: Sums all values in Region_Revenue column into a single float scalar.
    total_grand_revenue = float(regional_summary["Region_Revenue"].sum())

    # What is used: Vectorized Series division and scalar multiplication.
    # Why it is used: Calculates percentage contribution for each region.
    # How it works: Divides each region's revenue by grand total and multiplies by 100.
    if total_grand_revenue > 0:
        regional_summary["Percentage_Contribution"] = (
            (regional_summary["Region_Revenue"] / total_grand_revenue) * 100.0
        ).round(2)
    else:
        regional_summary["Percentage_Contribution"] = 0.0

    return regional_summary


if __name__ == "__main__":
    # What is used: Mock regional sales dataset.
    # Why it is used: Input data for regional percentage contribution analysis.
    # How it works: Maps regional revenue records into DataFrame.
    sales_df = pd.DataFrame({
        "Region": ["West", "East", "West", "South", "North", "East"],
        "Revenue": [80000, 50000, 60000, 90000, 40000, 30000]
    })

    # What is used: Calling calculate_regional_revenue_percentage.
    # Why it is used: Computes regional revenue breakdown and percentage contributions.
    # How it works: Prints regional revenue percentage contribution table.
    pct_summary = calculate_regional_revenue_percentage(sales_df)
    print("--- Regional Revenue Percentage Contribution ---")
    print(pct_summary)
