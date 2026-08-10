# ==============================================================================
# Program    : Typed List Average Function (Task 2)
# Objective  : Calculate numerical average of float list with explicit collection type hints.
# Concept    : Generic Collection Type Annotations (list[float])
# Why Used   : Restricts input container expectations to list of floats.
# ==============================================================================

# What is used : list[float] parameter type hint and float return type hint
# Why it is used: Mandates a list containing float or numeric elements returning a float
def average(numbers: list[float]) -> float:
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

def main() -> None:
    print("=== TASK 2: TYPED LIST AVERAGE DEMO ===")
    sample_scores: list[float] = [85.5, 90.0, 78.5, 92.0, 88.0]
    
    avg_result: float = average(sample_scores)
    print(f"Input Scores : {sample_scores}")
    print(f"Calculated Average Score : {avg_result:.2f}")

if __name__ == "__main__":
    main()
