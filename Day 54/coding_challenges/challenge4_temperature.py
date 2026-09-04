"""
===============================================================================
DAY 54 — CODING CHALLENGE 4: TEMPERATURE DATASET ANALYTICS
===============================================================================
Topic: 1D Array Statistics and Boolean Filtering
Goal: Given temperatures [28, 31, 29, 35, 33, 27, 30], compute metrics.
===============================================================================
"""

import numpy as np


def analyze_temperatures(temps: np.ndarray):
    """Compute temperature metrics: mean, max, min, days > 30, and temperature range."""
    # What is used: NumPy statistical functions and boolean mask counting.
    # Why it is used: Analyzes time-series scalar metrics without manual loops.
    # How it works: Evaluates mean, max, min, mask count (temps > 30), and max - min.
    avg_temp = np.mean(temps)
    max_temp = np.max(temps)
    min_temp = np.min(temps)
    days_above_30 = np.sum(temps > 30)
    temp_range = max_temp - min_temp

    return {
        "mean": avg_temp,
        "max": max_temp,
        "min": min_temp,
        "days_above_30": days_above_30,
        "temp_range": temp_range,
    }


if __name__ == "__main__":
    temperatures = np.array([28, 31, 29, 35, 33, 27, 30])
    res = analyze_temperatures(temperatures)
    print("Temperature Dataset Metrics:")
    for k, v in res.items():
        print(f"  {k:<15}: {v}")

    assert res["max"] == 35, "Max temp failed"
    assert res["min"] == 27, "Min temp failed"
    assert res["days_above_30"] == 3, "Days above 30 failed (31, 35, 33)"
    assert res["temp_range"] == 8, "Range failed (35 - 27)"
    print("[OK] Challenge 4 Passed Successfully!")
