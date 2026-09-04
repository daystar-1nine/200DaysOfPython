"""
Day 60 - Pure Python Challenge 2: Frequency Counter
Uses collections.Counter to determine frequency counts of products.
"""

# What is used: Import sys module and Counter from collections.
# Why it is used: Configures console output and performs element frequency counting.
# How it works: Brings sys and Counter into scope.
import sys
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def count_product_frequencies(products: list[str]) -> Counter:
    """
    Count product occurrences using collections.Counter.

    Args:
        products: List of product string names.

    Returns:
        Counter: Counter mapping product names to integer counts.
    """
    # What is used: Counter(products).
    # Why it is used: Efficiently counts hashable elements in O(N) time complexity.
    # How it works: Iterates through product list and tallies occurrences in a dictionary subclass.
    return Counter(products)


def main() -> None:
    products = [
        "Laptop",
        "Phone",
        "Laptop",
        "Tablet",
        "Phone",
        "Laptop"
    ]
    freq = count_product_frequencies(products)

    print("==================================================")
    print("            PRODUCT FREQUENCY COUNTER             ")
    print("==================================================")
    for prod, count in freq.most_common():
        print(f"Product: {prod:<10} | Frequency: {count}")


if __name__ == "__main__":
    main()
