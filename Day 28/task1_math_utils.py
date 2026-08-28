# ==============================================================================
# Module     : Math Utilities Module (Task 1)
# Objective  : Define basic math helper functions in an isolated module file.
# Concept    : Python Module Definition
# Why Used   : Provides reusable add and subtract functions for external imports.
# ==============================================================================

def add(a: float, b: float) -> float:
    """Returns the sum of a and b."""
    return a + b

def subtract(a: float, b: float) -> float:
    """Returns the difference of a and b."""
    return a - b

if __name__ == "__main__":
    print(f"[Module Test] add(10, 20) = {add(10, 20)}")
