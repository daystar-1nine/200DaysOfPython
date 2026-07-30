# ==============================================================================
# Program    : Generate Prime Numbers Using Generator
# Objective  : Stream prime numbers up to a specified limit.
# Concept    : Generator Functions & Primality Testing
# Why Used   : Evaluates and yields prime numbers lazily without computing all primes upfront.
# ==============================================================================

def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

# What is used : Generator function 'def prime_generator(limit)'
# Why it is used: Yields prime numbers on demand
def prime_generator(limit):
    for candidate in range(2, limit + 1):
        if is_prime(candidate):
            # What is used : yield keyword
            yield candidate

def main():
    print("=== Prime Numbers Generator (Up to 30) ===")
    primes = prime_generator(30)
    for p in primes:
        print(p, end=" ")
    print()

if __name__ == "__main__":
    main()
