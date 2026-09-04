"""
===============================================================================
DAY 54 — EXERCISE 1: PYTHON LOOP VS NUMPY VECTORIZATION BENCHMARK
===============================================================================
Topic: Computational Performance Comparison
Goal: Compare execution time of pure Python loop vs NumPy vectorized squaring
      across 1,000,000 numerical elements.
===============================================================================
"""

import time
import numpy as np


def run_performance_benchmark(n_elements: int = 1_000_001):
    """Benchmark pure Python list loop vs NumPy vectorized squaring operation."""
    # What is used: time.perf_counter() microsecond timer.
    # Why it is used: Demonstrates numerical speedup of compiled C vectorization over Python.
    # How it works: Measures execution seconds of Python for-loop vs NumPy arr ** 2.
    print(f"[INFO] Running Vectorization Benchmark over {n_elements - 1:,} elements...\n")

    # 1. Pure Python Loop
    start_py = time.perf_counter()
    py_list = list(range(1, n_elements))
    py_squares = [x ** 2 for x in py_list]
    end_py = time.perf_counter()
    py_time = end_py - start_py

    # 2. NumPy Vectorized Operation
    start_np = time.perf_counter()
    np_arr = np.arange(1, n_elements)
    np_squares = np_arr ** 2
    end_np = time.perf_counter()
    np_time = end_np - start_np

    speedup = py_time / np_time if np_time > 0 else 0.0

    print(f"  Pure Python List Execution Time: {py_time:.4f} seconds")
    print(f"  NumPy Vectorized Execution Time: {np_time:.4f} seconds")
    print(f"  Speedup Factor:                  {speedup:.2f}x Faster!")

    return py_time, np_time, speedup


if __name__ == "__main__":
    py_t, np_t, speedup = run_performance_benchmark(1_000_001)
    assert np_t < py_t, "NumPy vectorized operation should execute faster than Python list loop"
    print("\n[OK] Exercise 1 Vectorization Benchmark Passed Successfully!")
