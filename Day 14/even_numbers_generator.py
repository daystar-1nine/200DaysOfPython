# ==============================================================================
# Program    : Generator Function for Even Numbers
# Objective  : Generate even numbers within a given range lazily using yield.
# Concept    : Generator Functions & yield Keyword
# Why Used   : Produces numbers one at a time on demand without storing the entire sequence in memory.
# ==============================================================================

# What is used : Generator function 'def even_generator(start, limit)'
# Why it is used: Yields even numbers sequentially using yield keyword
# How it works : Pauses execution at yield, preserving local state for subsequent calls
def even_generator(start, limit):
    # Adjust start to first even number if odd
    if start % 2 != 0:
        start += 1

    current = start
    while current <= limit:
        # What is used : yield statement
        # Why it is used: Returns current value and pauses generator execution state
        yield current
        current += 2

def main():
    print("=== Even Numbers Generator (1 to 20) ===")
    
    # What is used : Generator object instantiation
    gen = even_generator(1, 20)

    # What is used : Iterating over generator object in a for loop
    # How it works : Automatically invokes next(gen) until StopIteration is raised
    for num in gen:
        print(num, end=" ")
    print()

if __name__ == "__main__":
    main()
