# ==============================================================================
# Program    : Generate Fibonacci Numbers Using Generator
# Objective  : Stream Fibonacci sequence up to N terms lazily.
# Concept    : State Preserving Generators (a, b = b, a + b)
# Why Used   : Generates Fibonacci values infinitely/dynamically with O(1) memory complexity.
# ==============================================================================

# What is used : Generator function 'def fibonacci_gen(n_terms)'
# Why it is used: Yields Fibonacci values 1-by-1
# How it works : Maintains state of previous two terms (a, b) across yields
def fibonacci_gen(n_terms):
    a, b = 0, 1
    count = 0
    while count < n_terms:
        # What is used : yield keyword
        # How it works : Suspends stack frame and yields value 'a' to consumer
        yield a
        # What is used : Tuple unpacking for state transition
        a, b = b, a + b
        count += 1

def main():
    print("=== Fibonacci Series Generator (First 10 Terms) ===")
    fib = fibonacci_gen(10)

    # What is used : Iterating over generator stream
    for term in fib:
        print(term, end=" ")
    print()

if __name__ == "__main__":
    main()
