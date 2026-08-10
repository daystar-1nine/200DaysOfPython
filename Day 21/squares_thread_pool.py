# ==============================================================================
# Program    : Calculate Squares using ThreadPoolExecutor (Task 4)
# Objective  : Compute squares of numbers 1 to 20 using ThreadPoolExecutor.
# Concept    : concurrent.futures.ThreadPoolExecutor
# Why Used   : Simplifies parallel function mapping and result aggregation across worker threads.
# ==============================================================================

from concurrent.futures import ThreadPoolExecutor

def calculate_square(n):
    """Returns square of integer n."""
    return n * n

def main():
    print("=== TASK 4: THREAD POOL EXECUTOR SQUARES (1-20) ===")
    numbers = list(range(1, 21))

    # What is used : ThreadPoolExecutor(max_workers=4)
    # Why it is used: Allocates numbers list mapping across 4 reusable worker threads
    with ThreadPoolExecutor(max_workers=4) as executor:
        squares = list(executor.map(calculate_square, numbers))

    print("Input Numbers:", numbers)
    print("Output Squares:", squares)

if __name__ == "__main__":
    main()
