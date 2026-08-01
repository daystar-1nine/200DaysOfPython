# ==============================================================================
# Program    : Employee Salary Processor (Challenge Project)
# Objective  : Process employee dictionary records functionally (map, filter, reduce, dict comp).
# Concept    : Functional Data Processing Pipelines on Dictionaries
# Why Used   : Increases salaries, filters high earners, sums total expense, builds dict map.
# ==============================================================================

from functools import reduce

employees = [
    {"name": "Amit", "salary": 40000},
    {"name": "Suraj", "salary": 55000},
    {"name": "Priya", "salary": 65000},
    {"name": "Rahul", "salary": 48000},
    {"name": "Neha", "salary": 72000}
]

def main():
    print("=== EMPLOYEE SALARY PROCESSOR ===")

    # 1. Increase salary by 10% for all employees using map()
    # What is used : map() with dictionary copy transformation
    updated_employees = list(map(lambda emp: {"name": emp["name"], "salary": round(emp["salary"] * 1.10, 2)}, employees))
    print("\n--- 1. Employees with 10% Salary Increment ---")
    for emp in updated_employees:
        print(f"Name: {emp['name']:<10} | Updated Salary: Rs.{emp['salary']:,.2f}")

    # 2. Filter employees earning above Rs.50,000 using filter()
    # What is used : filter() checking emp["salary"] > 50000
    high_earners = list(filter(lambda emp: emp["salary"] > 50000, updated_employees))
    print("\n--- 2. High Earners (Salary > Rs.50,000) ---")
    for emp in high_earners:
        print(f"Name: {emp['name']:<10} | Salary: Rs.{emp['salary']:,.2f}")

    # 3. Calculate total salary expense using reduce()
    # What is used : reduce() summing emp["salary"]
    total_payroll = reduce(lambda acc, emp: acc + emp["salary"], updated_employees, 0.0)
    print(f"\n3. Total Monthly Payroll Expense (reduce): Rs.{total_payroll:,.2f}")

    # 4. Create dictionary mapping employee names to updated salaries using Dict Comprehension
    # What is used : Dictionary Comprehension
    salary_dict = {emp["name"]: emp["salary"] for emp in updated_employees}
    print(f"\n4. Salary Lookup Map (Dict Comp):\n{salary_dict}")

if __name__ == "__main__":
    main()
