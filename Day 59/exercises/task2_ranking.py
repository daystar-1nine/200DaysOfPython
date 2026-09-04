"""
Day 59 - Exercise 2: Intra-Department Student Ranking
Demonstrates ranking student marks within department partitions using groupby().rank().
"""

# What is used: Import sys and pandas modules.
# Why it is used: Configures UTF-8 console output and performs group ranking operations.
# How it works: Brings sys and pandas namespaces into scope.
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    # What is used: Sample student DataFrame creation.
    # Why it is used: Provides dummy department marks data for rank evaluation.
    # How it works: Initializes DataFrame with Student, Department, and Marks.
    data = {
        "Student": ["Aarav", "Priya", "Kabir", "Neha", "Rohan", "Diya"],
        "Department": ["CS", "CS", "CS", "ECE", "ECE", "ECE"],
        "Marks": [85, 92, 85, 88, 95, 78]
    }
    df = pd.DataFrame(data)

    # What is used: groupby("Department")["Marks"].rank(ascending=False, method="dense").
    # Why it is used: Computes student rank order within their department partition.
    # How it works: Assigns rank 1 to highest mark, handling ties using dense method.
    df["Department_Rank"] = (
        df.groupby("Department")["Marks"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    # What is used: df.sort_values().
    # Why it is used: Sorts output cleanly by Department and Department_Rank.
    # How it works: Orders rows by Department alphabetically and Department_Rank ascending.
    sorted_df = df.sort_values(by=["Department", "Department_Rank"])

    print("--- Intra-Department Student Marks & Ranks ---")
    print(sorted_df)


if __name__ == "__main__":
    main()
