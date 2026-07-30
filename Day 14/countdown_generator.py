# ==============================================================================
# Program    : Countdown Generator
# Objective  : Generate a countdown sequence from N down to 0.
# Concept    : Decrementing Generators
# Why Used   : Yields decrementing values lazily.
# ==============================================================================

# What is used : Generator function 'def countdown(n)'
# Why it is used: Yields numbers from n down to 0 sequentially
def countdown(n):
    current = n
    while current >= 0:
        # What is used : yield keyword
        yield current
        current -= 1

def main():
    print("=== Countdown Generator (From 5 to 0) ===")
    timer = countdown(5)
    for t in timer:
        print(f"T-minus {t}...")
    print("[LAUNCH] Blast off!")

if __name__ == "__main__":
    main()
