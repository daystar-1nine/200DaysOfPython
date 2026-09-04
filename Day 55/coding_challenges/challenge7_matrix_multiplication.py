"""
===============================================================================
DAY 55 — CODING CHALLENGE 7: MATRIX MULTIPLICATION
===============================================================================
Topic: Linear Algebra Matrix Multiplication using @ Operator
Goal: Given A = [[1,2],[3,4]] and B = [[5,6],[7,8]], compute A @ B.
===============================================================================
"""

import numpy as np


def multiply_matrices(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    """Perform matrix multiplication of two 2D matrices using @ operator."""
    # What is used: Matrix multiplication @ operator (or np.matmul).
    # Why it is used: Computes matrix product efficiently in C.
    # How it works: Evaluates dot product of rows of A with columns of B.
    product = matrix_a @ matrix_b
    return product


if __name__ == "__main__":
    A = np.array([
        [1, 2],
        [3, 4]
    ])
    B = np.array([
        [5, 6],
        [7, 8]
    ])

    C = multiply_matrices(A, B)
    print("Matrix A:\n", A)
    print("Matrix B:\n", B)
    print("Product C (A @ B):\n", C)

    expected = np.array([
        [19, 22],
        [43, 50]
    ])
    assert np.array_equal(C, expected), f"Expected\n{expected}\ngot\n{C}"
    print("[OK] Challenge 7 Passed Successfully!")
