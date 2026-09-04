"""
===============================================================================
DAY 54 — CODING CHALLENGE 1: ARRAY BASICS & STATISTICAL METRICS
===============================================================================
Topic: 1D Array Generation and Basic Statistical Aggregations
Goal: Create a NumPy array of numbers from 1 to 100 and compute statistical summary.
===============================================================================
"""

import numpy as np


def analyze_array_basics():
    """Generate array 1..100 and compute statistical summary metrics."""
    # What is used: np.arange() and statistical aggregate functions.
    # Why it is used: Demonstrates vector array generation and statistical aggregation.
    # How it works: Computes sum, mean, median, std, min, and max across 1..100 array.
    numbers = np.arange(1, 101)

    total_sum = np.sum(numbers)
    mean_val = np.mean(numbers)
    median_val = np.median(numbers)
    std_val = np.std(numbers)
    min_val = np.min(numbers)
    max_val = np.max(numbers)

    return {
        "size": numbers.size,
        "sum": total_sum,
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
    }


if __name__ == "__main__":
    res = analyze_array_basics()
    print("NumPy Array Basics (1..100) Results:")
    for k, v in res.items():
        print(f"  {k:<10}: {v}")

    assert res["size"] == 100, "Size failed"
    assert res["sum"] == 5050, "Sum failed"
    assert res["mean"] == 50.5, "Mean failed"
    assert res["median"] == 50.5, "Median failed"
    assert res["min"] == 1, "Min failed"
    assert res["max"] == 100, "Max failed"
    print("[OK] Challenge 1 Passed Successfully!")
