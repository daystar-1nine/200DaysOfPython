# ==============================================================================
# Program    : Even Number Iterator (EvenNumberIterator)
# Objective  : Generate even numbers starting from 2 up to limit.
# Concept    : Mathematical Stream Iterator
# Why Used   : Demonstrates generating calculated sequence items dynamically.
# ==============================================================================

class EvenNumberIterator:
    def __init__(self, limit: int):
        if limit < 2:
            raise ValueError("Limit must be at least 2.")
        self.limit = limit
        self.current = 2

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self.current > self.limit:
            raise StopIteration
        val = self.current
        self.current += 2
        return val


if __name__ == "__main__":
    print("=== EVEN NUMBER ITERATOR DEMO ===")
    for even in EvenNumberIterator(10):
        print(f"Even: {even}")
