# ==============================================================================
# Program    : Expense Calculation Module (Task 4)
# Objective  : Calculate total amount from a list of expense dictionaries.
# Concept    : Pure Business Logic Function
# Why Used   : Provides pure function for testing empty, single, and multiple expense scenarios.
# ==============================================================================

def calculate_total(expenses: list[dict]) -> float:
    return sum(float(expense.get("amount", 0.0)) for expense in expenses)
