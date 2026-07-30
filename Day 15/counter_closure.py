# ==============================================================================
# Program    : Counter Stateful Closure
# Objective  : Maintain a stateful counter across invocations using `nonlocal`.
# Concept    : Closures with `nonlocal` State Mutation
# Why Used   : `nonlocal` allows inner closure functions to mutate enclosing scope variables.
# ==============================================================================

# What is used : Outer factory function 'def create_counter(start=0)'
def create_counter(start=0):
    count = start

    # What is used : Inner closure function
    def increment(step=1):
        # What is used : 'nonlocal count' keyword
        # Why it is used: Declares intent to modify 'count' variable in enclosing outer function scope
        # How it works : Prevents Python from creating a new local variable named count
        nonlocal count
        count += step
        return count

    return increment

def main():
    print("=== Counter Closure Demonstration ===")
    counter = create_counter(start=100)

    print("Call 1 (step 1)  :", counter())
    print("Call 2 (step 5)  :", counter(5))
    print("Call 3 (step 10) :", counter(10))

if __name__ == "__main__":
    main()
