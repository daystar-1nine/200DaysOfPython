# ==============================================================================
# Program    : Countdown Iterator (CountdownIterator)
# Objective  : Produce countdown integers from start down to 1.
# Concept    : Custom State-Based Iterator Protocol
# Why Used   : Demonstrates StopIteration on reaching zero bound.
# ==============================================================================

class CountdownIterator:
    def __init__(self, start: int):
        if start < 1:
            raise ValueError("Start count must be at least 1.")
        # What is used : State Pointer Initialization
        # Why it is used: Tracks remaining count during iteration
        self.current = start

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self.current <= 0:
            raise StopIteration
        val = self.current
        self.current -= 1
        return val


if __name__ == "__main__":
    print("=== COUNTDOWN ITERATOR DEMO ===")
    for num in CountdownIterator(5):
        print(f"Countdown: {num}")
