"""
Day 60 - Pure Python Challenge 4: Employee Dataclass & Aggregation
Creates Employee dataclass and calculates average salary, highest salary, and department employee counts.
"""

# What is used: Import sys module, dataclass from dataclasses, and Counter from collections.
# Why it is used: Domain modeling and pure Python metric aggregations.
# How it works: Brings dataclass and Counter into module scope.
import sys
from collections import Counter
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# What is used: @dataclass decorator.
# Why it is used: Encapsulates employee entity without manual __init__ or __repr__ methods.
# How it works: Generates pythonic class structure for name, department, and salary.
@dataclass
class Employee:
    name: str
    department: str
    salary: float


def compute_employee_metrics(employees: list[Employee]) -> dict:
    """
    Compute average salary, highest salary, and department counts for employee list.

    Args:
        employees: List of Employee dataclass instances.

    Returns:
        dict: Aggregated employee metrics dictionary.
    """
    if not employees:
        return {"average_salary": 0.0, "highest_salary": 0.0, "top_earner": "N/A", "department_counts": {}}

    salaries = [e.salary for e in employees]
    avg_sal = round(sum(salaries) / len(salaries), 2)
    max_sal = max(salaries)
    top_earner = [e.name for e in employees if e.salary == max_sal][0]

    dept_counts = dict(Counter(e.department for e in employees))

    return {
        "average_salary": avg_sal,
        "highest_salary": max_sal,
        "top_earner": top_earner,
        "department_counts": dept_counts
    }


def main() -> None:
    employees = [
        Employee("Rahul Sharma", "Engineering", 85000.0),
        Employee("Priya Patel", "Data Science", 92000.0),
        Employee("Aman Verma", "Marketing", 55000.0),
        Employee("Sneha Kulkarni", "Engineering", 98000.0),
        Employee("Vikram Singh", "Data Science", 78000.0)
    ]

    metrics = compute_employee_metrics(employees)

    print("==================================================")
    print("          EMPLOYEE DATACLASS ANALYTICS            ")
    print("==================================================")
    print(f"Total Employees  : {len(employees)}")
    print(f"Average Salary   : ₹{metrics['average_salary']:,.2f}")
    print(f"Highest Salary   : ₹{metrics['highest_salary']:,.2f} ({metrics['top_earner']})")
    print(f"Department Counts: {metrics['department_counts']}")


if __name__ == "__main__":
    main()
