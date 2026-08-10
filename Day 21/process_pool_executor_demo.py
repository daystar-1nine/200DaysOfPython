# ==============================================================================
# Program    : ProcessPoolExecutor Parallel Execution
# Objective  : Execute CPU-intensive functions in parallel using ProcessPoolExecutor.
# Concept    : concurrent.futures.ProcessPoolExecutor
# Why Used   : Distributes workload across all available system CPU cores.
# ==============================================================================

from concurrent.futures import ProcessPoolExecutor
import time

def compute_heavy_math(n):
    """Calculates sum of squares up to n (CPU-bound)."""
    return sum(i * i for i in range(1, n + 1))

def main():
    print("=== PROCESS POOL EXECUTOR DEMO ===")
    numbers = [2_000_000, 3_000_000, 2_500_000, 3_500_000]

    start_time = time.time()

    # What is used : ProcessPoolExecutor() inside context manager
    # Why it is used: Automatically manages process pool sized to available CPU cores
    with ProcessPoolExecutor() as executor:
        # What is used : executor.map(compute_heavy_math, numbers)
        results = list(executor.map(compute_heavy_math, numbers))

    elapsed = time.time() - start_time
    print("Heavy Math Calculation Results:", results)
    print(f"Parallel Process Pool Computation Time: {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
