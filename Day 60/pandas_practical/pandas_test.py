"""
Day 60 - Pandas Practical Test: Student DataFrame Operations
Executes all 10 standard Pandas practical test requirements.
"""

# What is used: Import sys and pandas library.
# Why it is used: Cross-platform output encoding and tabular DataFrame manipulation.
# How it works: Brings sys and pandas namespaces into scope.
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_pandas_practical_test() -> tuple[pd.DataFrame, dict]:
    """
    Execute all 10 Pandas practical operations on student dataset.

    Returns:
        tuple[pd.DataFrame, dict]: Final mutated DataFrame and results dictionary.
    """
    df = pd.DataFrame({
        "Name": ["A", "B", "C", "D", "E"],
        "Department": ["CSE", "DS", "CSE", "ECE", "DS"],
        "Age": [20, 21, 19, 22, 20],
        "Marks": [85, 92, 76, 68, 95]
    })

    # 1. Inspect properties
    shape = df.shape
    columns = list(df.columns)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # 2. Average marks
    avg_marks = round(float(df["Marks"].mean()), 2)

    # 3. Find the topper
    topper_row = df.loc[df["Marks"].idxmax()]
    topper = {"Name": topper_row["Name"], "Marks": topper_row["Marks"]}

    # 4. Filter marks > 80
    high_scorers = df[df["Marks"] > 80]

    # 5. Sort descending by marks
    sorted_df = df.sort_values(by="Marks", ascending=False)

    # 6. Add Result column
    df["Result"] = df["Marks"].apply(lambda m: "Pass" if m >= 50 else "Fail")

    # 7. Add Grade column
    def get_grade(m: int) -> str:
        if m >= 90:
            return "A+"
        elif m >= 80:
            return "A"
        elif m >= 70:
            return "B"
        return "C"

    df["Grade"] = df["Marks"].apply(get_grade)

    # 8. Average marks by department
    dept_avg = df.groupby("Department")["Marks"].mean().round(2).to_dict()

    # 9. Number of students per department
    dept_counts = df["Department"].value_counts().to_dict()

    # 10. Highest-scoring student in each department
    dept_toppers = (
        df.sort_values(by="Marks", ascending=False)
        .groupby("Department")
        .first()[["Name", "Marks"]]
        .to_dict(orient="index")
    )

    summary = {
        "shape": shape,
        "columns": columns,
        "dtypes": dtypes,
        "avg_marks": avg_marks,
        "topper": topper,
        "high_scorers_count": len(high_scorers),
        "dept_avg": dept_avg,
        "dept_counts": dept_counts,
        "dept_toppers": dept_toppers
    }

    return df, summary


def main() -> None:
    final_df, res = run_pandas_practical_test()

    print("==================================================")
    print("             PANDAS PRACTICAL EXAM RESULTS        ")
    print("==================================================")
    print(f"1. Shape: {res['shape']} | Columns: {res['columns']}")
    print(f"2. Average Marks: {res['avg_marks']}")
    print(f"3. Class Topper : {res['topper']['Name']} ({res['topper']['Marks']} Marks)")
    print(f"4. High Scorers (>80 Count): {res['high_scorers_count']}")
    print(f"8. Department Average Marks: {res['dept_avg']}")
    print(f"9. Department Student Count: {res['dept_counts']}")
    print(f"10. Highest Scorer per Dept: {res['dept_toppers']}")
    print("\n--- Final Enriched DataFrame ---")
    print(final_df)


if __name__ == "__main__":
    main()
