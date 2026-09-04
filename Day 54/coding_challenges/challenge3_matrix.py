"""
===============================================================================
DAY 54 — CODING CHALLENGE 3: 3x3 MATRIX ANALYTICS & TRANSPOSE
===============================================================================
Topic: 2D Matrix Manipulation, Transposition, and Axis Aggregations
Goal: Analyze a 3x3 matrix [[1,2,3],[4,5,6],[7,8,9]].
===============================================================================
"""

import numpy as np


def analyze_matrix_properties():
    """Analyze properties of a 3x3 2D matrix."""
    # What is used: 2D array reshape, transpose .T, and axis aggregations.
    # Why it is used: Demonstrates row-wise (axis=1) and column-wise (axis=0) aggregations.
    # How it works: Evaluates shape, transpose, sum, row sums, column sums, max, and min.
    matrix = np.arange(1, 10).reshape(3, 3)

    return {
        "shape": matrix.shape,
        "transpose": matrix.T,
        "total_sum": np.sum(matrix),
        "row_sums": np.sum(matrix, axis=1),
        "col_sums": np.sum(matrix, axis=0),
        "max": np.max(matrix),
        "min": np.min(matrix),
    }


if __name__ == "__main__":
    res = analyze_matrix_properties()
    print("3x3 Matrix Analysis Results:")
    print("  Shape:     ", res["shape"])
    print("  Total Sum: ", res["total_sum"])
    print("  Row Sums:  ", res["row_sums"])
    print("  Col Sums:  ", res["col_sums"])
    print("  Min / Max: ", res["min"], "/", res["max"])
    print("  Transpose:\n", res["transpose"])

    assert res["shape"] == (3, 3), "Shape failed"
    assert res["total_sum"] == 45, "Total sum failed"
    assert np.array_equal(res["row_sums"], np.array([6, 15, 24])), "Row sums failed"
    assert np.array_equal(res["col_sums"], np.array([12, 15, 18])), "Col sums failed"
    print("[OK] Challenge 3 Passed Successfully!")
