# ==============================================================================
# Program    : Fibonacci Generation Using Functional Tools (Advanced)
# Objective  : Generate Fibonacci sequence using reduce() and list comprehension.
# Concept    : Functional Sequence Accumulation via reduce()
# Why Used   : Demonstrates building list structures dynamically using reduce accumulators.
# ==============================================================================

from functools import reduce

# What is used : Function generating N terms of Fibonacci using reduce()
def get_fibonacci_series(n):
    if n <= 0:
        return []
    if n == 1:
        return [0]
    
    # What is used : reduce() with list accumulator
    # How it works : acc starts as [0, 1]; for each iteration, appends (acc[-1] + acc[-2])
    return reduce(lambda acc, _: acc + [acc[-1] + acc[-2]], range(n - 2), [0, 1])

def main():
    print("=== Fibonacci Series via Functional reduce() ===")
    n_terms = 10
    fib_list = get_fibonacci_series(n_terms)
    print(f"First {n_terms} Fibonacci Terms: {fib_list}")

if __name__ == "__main__":
    main()
