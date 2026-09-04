"""
===============================================================================
DAY 51 — CODING CHALLENGE 4: EMPLOYEE DATACLASS & COLLECTIONS ANALYTICS
===============================================================================
This module creates an Employee dataclass and computes highest salary, average salary,
and department grouping breakdown using collections.defaultdict.
===============================================================================
"""

from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Any


@dataclass
class Employee:
    """Dataclass representing employee organization details."""
    id: int
    name: str
    department: str
    salary: float


def analyze_employees(employees: List[Employee]) -> Dict[str, Any]:
    """Perform salary and department aggregation analytics on employee list."""
    # What is used: max() function for highest salary search.
    # Why it is used: Identifies the employee with maximum salary.
    # How it works: Compiles max over employees list by salary key.
    highest_emp = max(employees, key=lambda e: e.salary)

    # What is used: Generator sum calculation for average salary.
    # Why it is used: Computes mean salary across all employees.
    # How it works: Sums salaries and divides by total count.
    avg_salary = sum(e.salary for e in employees) / len(employees)

    # What is used: collections.defaultdict(list) for department grouping.
    # Why it is used: Automatically initializes list per department key.
    # How it works: Appends employee names to department list without key error checks.
    dept_map: Dict[str, List[str]] = defaultdict(list)
    for emp in employees:
        dept_map[emp.department].append(emp.name)

    return {
        "highest_salary": highest_emp,
        "average_salary": avg_salary,
        "department_map": dict(dept_map),
    }


if __name__ == "__main__":
    staff = [
        Employee(1, "Alice", "Engineering", 95000.0),
        Employee(2, "Bob", "HR", 60000.0),
        Employee(3, "Charlie", "Engineering", 110000.0),
        Employee(4, "Diana", "Marketing", 75000.0),
        Employee(5, "Evan", "HR", 65000.0),
    ]
    res = analyze_employees(staff)
    print("Highest Salary:", res["highest_salary"].name, res["highest_salary"].salary)
    print("Average Salary:", res["average_salary"])
    print("Department Map:", res["department_map"])

    assert res["highest_salary"].name == "Charlie"
    assert res["average_salary"] == 81000.0
    assert len(res["department_map"]["Engineering"]) == 2
    print("[OK] Challenge 4 Passed!")
