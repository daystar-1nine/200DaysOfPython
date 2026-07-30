# ==============================================================================
# Program    : Infinite Number Generator
# Objective  : Stream infinite sequence of numbers with manual break control.
# Concept    : Infinite Generators (while True + yield)
# Why Used   : Generates an endless stream of auto-incrementing numbers without memory limits.
# ==============================================================================

# What is used : Infinite Generator function 'def infinite_numbers(start=1)'
# Why it is used: Keeps generating incremented values endlessly until caller breaks
def infinite_numbers(start=1):
    current = start
    while True:
        # What is used : yield statement inside infinite loop
        yield current
        current += 1

def main():
    print("=== Infinite Number Generator (First 10 Generated) ===")
    gen = infinite_numbers(start=100)

    # Manual loop break after 10 items
    for _ in range(10):
        # What is used : next(gen)
        print(next(gen), end=" ")
    print("\n[Stopped infinite generator manually]")

if __name__ == "__main__":
    main()
