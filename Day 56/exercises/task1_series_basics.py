"""
Day 56 - Practical Task 1: Series Fundamentals
Demonstrates Pandas Series creation, custom indexing, scalar operations, and string accessor methods.
"""

# What is used: Import pandas library.
# Why it is used: Fundamental dependency for Series creation and element manipulation.
# How it works: Brings pandas package into execution context.
import pandas as pd


def demonstrate_series_basics() -> dict:
    """
    Demonstrate Series creation, custom index, indexing, and statistics.

    Returns:
        dict: Collection of processed Pandas Series objects.
    """
    # What is used: pd.Series constructor with custom index.
    # Why it is used: Creates a 1D labelled array of student scores.
    # How it works: Binds data list [85, 92, 78, 90] to string index labels.
    scores = pd.Series([85, 92, 78, 90], index=["Alice", "Bob", "Charlie", "David"], name="Math_Score")

    # What is used: Series label indexing scores["Bob"].
    # Why it is used: Retrieves score for Bob using key lookup.
    # How it works: Searches index hash map and returns corresponding scalar value (92).
    bob_score = scores["Bob"]

    # What is used: Vectorized scalar addition scores + 5.
    # Why it is used: Adds bonus points to all students efficiently without explicit loops.
    # How it works: Adds 5 to every element in the underlying NumPy array.
    bonus_scores = scores + 5

    # What is used: Series aggregate functions mean(), max(), std().
    # Why it is used: Summarizes statistical properties of the Series.
    # How it works: Computes mean, maximum, and standard deviation across elements.
    stats = {
        "mean": float(scores.mean()),
        "max": float(scores.max()),
        "std": float(scores.std())
    }

    return {
        "scores": scores,
        "bob_score": bob_score,
        "bonus_scores": bonus_scores,
        "stats": stats
    }


if __name__ == "__main__":
    results = demonstrate_series_basics()
    print("--- Original Scores ---")
    print(results["scores"])
    print(f"\nBob's Score: {results['bob_score']}")
    print("\n--- Scores with 5 Bonus Points ---")
    print(results["bonus_scores"])
    print("\n--- Summary Statistics ---")
    print(results["stats"])
