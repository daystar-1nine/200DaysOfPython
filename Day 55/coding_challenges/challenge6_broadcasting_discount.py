"""
===============================================================================
DAY 55 — CODING CHALLENGE 6: BROADCASTING DISCOUNT APPLICATION
===============================================================================
Topic: Multidimensional Array Broadcasting
Goal: Apply 1D discount vector [10, 20, 30] to 2D prices matrix [[100,200,300],[150,250,350]].
===============================================================================
"""

import numpy as np


def apply_discount_broadcasting(prices: np.ndarray, discount: np.ndarray) -> np.ndarray:
    """Subtract row-wise discount vector from 2D prices matrix via broadcasting."""
    # What is used: NumPy broadcasting subtraction prices - discount.
    # Why it is used: Applies 1D vector across all rows of 2D matrix without nested loops.
    # How it works: Aligns (2,3) matrix with (3,) vector, subtracting discounts element-wise per column.
    discounted_prices = prices - discount
    return discounted_prices


if __name__ == "__main__":
    prices = np.array([
        [100, 200, 300],
        [150, 250, 350]
    ])
    discount = np.array([10, 20, 30])

    result = apply_discount_broadcasting(prices, discount)
    print("Raw Prices Matrix:\n", prices)
    print("Discount Vector:   ", discount)
    print("Discounted Prices Matrix:\n", result)

    expected = np.array([
        [90, 180, 270],
        [140, 230, 320]
    ])
    assert np.array_equal(result, expected), f"Expected\n{expected}\ngot\n{result}"
    print("[OK] Challenge 6 Passed Successfully!")
