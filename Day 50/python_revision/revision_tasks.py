"""
===============================================================================
DAY 50 — PYTHON CORE REVISION IMPLEMENTATIONS
===============================================================================
This module consolidates key core Python concepts from Days 1 to 40, including
profile formatting, grade system validation, pattern generation, modular
calculator functions, dictionary management, range comprehensions, safe exception
handling, file-based persistence, and object-oriented BankAccount management.
===============================================================================
"""

import os
from typing import Dict, List, Any


def create_user_profile(name: str, age: int, college: str, percentage: float) -> str:
    """Format user details into a structured profile string."""
    # What is used: f-string string interpolation formatting.
    # Why it is used: Combines variable values into a multi-line output string cleanly.
    # How it works: Evaluates expressions inside curly braces {} at runtime.
    profile = (
        f"=== USER PROFILE ===\n"
        f"Name       : {name}\n"
        f"Age        : {age}\n"
        f"College    : {college}\n"
        f"Percentage : {percentage:.2f}%\n"
        f"===================="
    )
    return profile


def calculate_grade(marks: float) -> str:
    """Calculate student grade with input validation bounds [0, 100]."""
    # What is used: Range conditional boundary checking.
    # Why it is used: Ensures input marks are within valid percentage limits.
    # How it works: Raises ValueError if marks are negative or exceed 100.
    if marks < 0 or marks > 100:
        raise ValueError("Marks must be between 0 and 100 inclusive.")

    # What is used: Multi-branch conditional evaluation (if/elif/else).
    # Why it is used: Maps numeric ranges to academic letter grades.
    # How it works: Evaluates conditions sequentially from top to bottom.
    if marks >= 90:
        return "A"
    elif marks >= 80:
        return "B"
    elif marks >= 70:
        return "C"
    elif marks >= 60:
        return "D"
    else:
        return "F"


def generate_pattern_and_table(n: int) -> Dict[str, Any]:
    """Generate numeric pyramid string and multiplication table list."""
    # What is used: List comprehension with string repetition.
    # Why it is used: Builds repeated digit lines for pattern printing.
    # How it works: Multiplies digit string by row integer i.
    pyramid = [str(i) * i for i in range(1, n + 1)]

    # What is used: List comprehension for multiplication table.
    # Why it is used: Computes product values for multiplier range 1 to 10.
    # How it works: Iterates i from 1 to 10 and formats product string.
    table = [f"{n} x {i} = {n * i}" for i in range(1, 11)]

    return {"pyramid": pyramid, "table": table}


def calculator_add(a: float, b: float) -> float:
    """Addition utility."""
    return a + b


def calculator_subtract(a: float, b: float) -> float:
    """Subtraction utility."""
    return a - b


def calculator_multiply(a: float, b: float) -> float:
    """Multiplication utility."""
    return a * b


def calculator_divide(a: float, b: float) -> float:
    """Division utility with zero check."""
    # What is used: Explicit division by zero check.
    # Why it is used: Prevents runtime ZeroDivisionError exception.
    # How it works: Raises ZeroDivisionError with explicit error message if b is 0.
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero.")
    return a / b


class StudentDictionaryManager:
    """In-memory student records collection manager using dict of dicts."""

    def __init__(self) -> None:
        # What is used: Type-annotated nested dictionary initialization.
        # Why it is used: Stores student records indexed by unique student ID strings.
        # How it works: Initializes empty dictionary container.
        self.students: Dict[str, Dict[str, Any]] = {
            "101": {"name": "Rahul", "marks": 85.0},
            "102": {"name": "Aisha", "marks": 92.0},
        }

    def add_student(self, student_id: str, name: str, marks: float) -> None:
        """Add new student record."""
        # What is used: Dictionary key lookup and assignment.
        # Why it is used: Inserts or updates student entry in collection.
        # How it works: Maps student_id key to dictionary payload.
        self.students[student_id] = {"name": name, "marks": marks}

    def remove_student(self, student_id: str) -> bool:
        """Remove student by ID."""
        # What is used: Dict pop method with None fallback.
        # Why it is used: Safely removes student key if present without throwing KeyError.
        # How it works: Returns removed record or None if key missing.
        return self.students.pop(student_id, None) is not None

    def search_student(self, student_id: str) -> Dict[str, Any] | None:
        """Search student by ID."""
        # What is used: Dict get method.
        # Why it is used: Fetches student details safely.
        # How it works: Returns dict record if found, else None.
        return self.students.get(student_id)

    def calculate_average(self) -> float:
        """Calculate average marks across all students."""
        # What is used: Sum built-in with generator expression.
        # Why it is used: Computes total sum of student marks efficiently.
        # How it works: Iterates values in self.students and averages sum over length.
        if not self.students:
            return 0.0
        total = sum(s["marks"] for s in self.students.values())
        return total / len(self.students)


def generate_range_comprehensions(limit: int = 100) -> Dict[str, List[int]]:
    """Demonstrate list comprehensions over a range of natural numbers."""
    # What is used: Filtering list comprehensions.
    # Why it is used: Categorizes numbers based on modulus properties into lists.
    # How it works: Evaluates if condition for each element in range(1, limit + 1).
    evens = [x for x in range(1, limit + 1) if x % 2 == 0]
    odds = [x for x in range(1, limit + 1) if x % 2 != 0]
    squares = [x ** 2 for x in range(1, 11)]
    divisible_by_5 = [x for x in range(1, limit + 1) if x % 5 == 0]

    return {
        "evens": evens,
        "odds": odds,
        "squares": squares,
        "divisible_by_5": divisible_by_5,
    }


def safe_calculator_execute(op: str, num1_raw: str, num2_raw: str) -> float:
    """Execute arithmetic operation safely handling input conversion errors."""
    # What is used: Try-except exception handling block.
    # Why it is used: Intercepts ValueError and ZeroDivisionError gracefully.
    # How it works: Converts string inputs to floats and dispatches operator.
    try:
        a = float(num1_raw)
        b = float(num2_raw)
        if op == "+":
            return calculator_add(a, b)
        elif op == "-":
            return calculator_subtract(a, b)
        elif op == "*":
            return calculator_multiply(a, b)
        elif op == "/":
            return calculator_divide(a, b)
        else:
            raise ValueError(f"Unsupported operation '{op}'.")
    except ValueError as e:
        raise ValueError(f"Invalid input: {e}")
    except ZeroDivisionError as e:
        raise ZeroDivisionError(f"Math error: {e}")


class BankAccount:
    """Encapsulated BankAccount class managing financial transactions."""

    def __init__(self, owner: str, initial_balance: float = 0.0) -> None:
        # What is used: Private instance attribute naming protocol (_balance).
        # Why it is used: Encapsulates account balance preventing direct external mutation.
        # How it works: Initializes owner and balance attributes.
        if initial_balance < 0:
            raise ValueError("Initial balance cannot be negative.")
        self.owner: str = owner
        self._balance: float = initial_balance

    @property
    def balance(self) -> float:
        """Read-only property for account balance."""
        return self._balance

    def deposit(self, amount: float) -> float:
        """Deposit funds into account."""
        # What is used: Input validation boundary check.
        # Why it is used: Prevents depositing zero or negative amounts.
        # How it works: Adds amount to self._balance and returns new total.
        if amount <= 0:
            raise ValueError("Deposit amount must be positive.")
        self._balance += amount
        return self._balance

    def withdraw(self, amount: float) -> float:
        """Withdraw funds from account."""
        # What is used: Insufficient funds validation check.
        # Why it is used: Enforces account balance non-negativity constraint.
        # How it works: Subtracts amount if amount <= self._balance else raises ValueError.
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive.")
        if amount > self._balance:
            raise ValueError("Insufficient funds for withdrawal.")
        self._balance -= amount
        return self._balance


if __name__ == "__main__":
    # Demonstration of revision tasks
    print(create_user_profile("Suraj Sawant", 21, "Engineering College", 88.5))
    print("Grade for 85:", calculate_grade(85))
    print("Pattern/Table:", generate_pattern_and_table(5))

    mgr = StudentDictionaryManager()
    mgr.add_student("103", "Karan", 78.0)
    print("Average Marks:", mgr.calculate_average())

    account = BankAccount("Suraj", 1000.0)
    account.deposit(500.0)
    account.withdraw(200.0)
    print(f"Final Balance for {account.owner}: ${account.balance:.2f}")
