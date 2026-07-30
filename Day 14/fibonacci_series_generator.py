# ==============================================================================
# Program    : Pure Fibonacci Series Generator (Challenge Project)
# Objective  : Generate Fibonacci sequence (0, 1, 1, 2, 3, 5, 8, 13, 21...) using yield only.
# Concept    : Pure Generator Functions & Tuple Unpacking State Transitions
# Why Used   : Uses only yield and a generator function for O(1) space complexity Fibonacci calculations.
# ==============================================================================

# What is used : Generator function 'def fibonacci_series_gen(limit_terms=10)'
# Why it is used: Yields terms (0, 1, 1, 2, 3, 5, 8, 13, 21...) using only yield
def fibonacci_series_gen(limit_terms=10):
    a, b = 0, 1
    count = 0
    while count < limit_terms:
        # What is used : yield statement
        # How it works : Pauses execution and yields current value of 'a'
        yield a
        # What is used : Tuple unpacking for state transition
        a, b = b, a + b
        count += 1

def main():
    print("=== Pure Fibonacci Series Generator ===")
    try:
        n = int(input("Enter number of terms to generate (e.g. 9): "))
        if n <= 0:
            print("Number of terms must be greater than zero!")
            return

        print(f"\nFibonacci Sequence ({n} terms):")
        # What is used : Generator invocation
        fib_gen = fibonacci_series_gen(n)
        for term in fib_gen:
            print(term)

    except ValueError:
        print("Input Error: Please enter a valid integer!")

if __name__ == "__main__":
    main()
