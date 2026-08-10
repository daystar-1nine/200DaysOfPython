# ==============================================================================
# Program    : CPU-Heavy Mathematical Operations with ProcessPoolExecutor (Task 5)
# Objective  : Execute CPU-bound prime count calculations in parallel using ProcessPoolExecutor.
# Concept    : Process-Level Parallelism & GIL Bypass
# Why Used   : ProcessPoolExecutor utilizes multiple physical CPU cores for heavy math loops.
# ==============================================================================

from concurrent.futures import ProcessPoolExecutor
import time

def count_primes_in_range(n):
    """Counts prime numbers up to n (CPU-heavy task)."""
    count = 0
    for num in range(2, n):
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            count += 1
    return count

def main():
    print("=== TASK 5: PROCESS POOL EXECUTOR CPU-HEAVY MATH ===")
    ranges = [50_000, 60_000, 70_000, 80_000]

    start_time = time.time()

    # What is used : ProcessPoolExecutor()
    # Why it is used: Runs heavy prime counting loops in parallel across distinct CPU process cores
    with ProcessPoolExecutor() as executor:
        prime_counts = list(executor.map(count_primes_in_range, ranges))

    total_time = time.time() - start_time
    print(f"Ranges Tested         : {ranges}")
    print(f"Prime Numbers Found   : {prime_counts}")
    print(f"Parallel Execution Time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
