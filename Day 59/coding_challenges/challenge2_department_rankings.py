"""
Day 59 - Coding Challenge 2: Multi-Method Intra-Department Ranking
Evaluates and compares different ranking methods ('dense', 'min', 'first') within department partitions.
"""

# What is used: Import sys and pandas modules.
# Why it is used: Configures UTF-8 console output and evaluates ranking methods.
# How it works: Brings sys and pandas namespaces into scope.
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def rank_employees_by_department(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank employee performance within department partitions using multiple tie-breaking methods.

    Args:
        df: Input DataFrame containing 'Department' and 'Performance_Score' columns.

    Returns:
        pd.DataFrame: DataFrame enriched with Dense_Rank, Min_Rank, and First_Rank.
    """
    result_df = df.copy()

    # What is used: groupby().rank() with methods 'dense', 'min', 'first'.
    # Why it is used: Demonstrates tie-handling strategies in intra-group ranking.
    # How it works: Computes group ranks descending by Performance_Score.
    group = result_df.groupby("Department")["Performance_Score"]
    result_df["Dense_Rank"] = group.rank(ascending=False, method="dense").astype(int)
    result_df["Min_Rank"] = group.rank(ascending=False, method="min").astype(int)
    result_df["First_Rank"] = group.rank(ascending=False, method="first").astype(int)

    return result_df.sort_values(by=["Department", "Dense_Rank"])


def main() -> None:
    data = {
        "Employee": ["Rahul", "Priya", "Aman", "Sneha", "Vikram", "Ananya"],
        "Department": ["Sales", "Sales", "Sales", "HR", "HR", "HR"],
        "Performance_Score": [90, 90, 80, 95, 88, 95]
    }
    df = pd.DataFrame(data)
    ranked = rank_employees_by_department(df)
    print("--- Intra-Department Employee Performance Rankings ---")
    print(ranked)


if __name__ == "__main__":
    main()
