# ==============================================================================
# Program    : Custom Number Generator (Mini Project)
# Objective  : Produce numbers starting from a start value, incrementing by step size up to a limit.
# Concept    : Parameterized Generator Functions
# Why Used   : Generates custom arithmetic progression sequences lazily (e.g. 5, 10, 15, 20...).
# ==============================================================================

# What is used : Custom Parameterized Generator 'def custom_step_gen(start, step, limit)'
# Why it is used: Yields progression values (start, start+step, start+2*step) up to limit
def custom_step_gen(start, step, limit):
    current = start
    while current <= limit:
        # What is used : yield statement
        yield current
        current += step

def main():
    print("=== Custom Number Generator ===")
    try:
        start_val = int(input("Enter Start Number (e.g. 5): "))
        step_val = int(input("Enter Step Size (e.g. 5): "))
        limit_val = int(input("Enter Limit (e.g. 50): "))

        if step_val <= 0:
            print("Step size must be a positive integer!")
            return

        print(f"\nGenerating sequence from {start_val} with step {step_val} up to {limit_val}:")

        # What is used : Instantiating generator object
        gen = custom_step_gen(start_val, step_val, limit_val)

        # What is used : Iterating over generator stream
        for num in gen:
            print(num)

    except ValueError:
        print("Input Error: Please enter valid integer inputs!")

if __name__ == "__main__":
    main()
