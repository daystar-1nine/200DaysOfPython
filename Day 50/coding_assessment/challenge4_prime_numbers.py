"""
===============================================================================
DAY 50 — CODING CHALLENGE 4: PRIME NUMBERS GENERATOR
===============================================================================
This module computes and returns all prime numbers within a specified range [start, end].
===============================================================================
"""

from typing import List

def is_prime(n: int) -> bool:
    """Check if a number n is prime using trial division up to sqrt(n)."""
    # What is used: Mathematical primality trial division up to sqrt(n).
    # Why it is used: Determines primality efficiently in O(sqrt(n)) time.
    # How it works: Checks if n < 2, then tests divisibility from 2 up to int(n**0.5) + 1.
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def get_primes(start: int, end: int) -> List[int]:
    """Return a list of all prime numbers between start and end inclusive."""
    # What is used: List comprehension with primality predicate filter.
    # Why it is used: Filters range elements cleanly in a single expression.
    # How it works: Evaluates is_prime(x) for each x in range(start, end + 1).
    return [x for x in range(start, end + 1) if is_prime(x)]


if __name__ == "__main__":
    primes = get_primes(1, 100)
    print(f"Primes (1 to 100): {primes}")
    expected_count = 25
    assert len(primes) == expected_count, f"Expected 25 primes, got {len(primes)}"
    assert primes[0] == 2 and primes[-1] == 97, "Challenge 4 bounds invalid!"
    print("✅ Challenge 4 Passed!")
