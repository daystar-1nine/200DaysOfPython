# ==============================================================================
# Program    : Employee Class Implementation
# Objective  : Model employee records and calculate annual compensation.
# Concept    : Encapsulation & Instance Method Computations
# Why Used   : Groups employee attributes (emp_id, name, salary, department) and calculates annual salary.
# ==============================================================================

# What is used : Class definition 'class Employee:'
class Employee:
    """Class representing an organization employee."""

    # What is used : Constructor __init__
    def __init__(self, emp_id, name, monthly_salary, department):
        self.emp_id = emp_id
        self.name = name
        self.salary = monthly_salary
        self.department = department

    # What is used : Instance method calculating annual compensation
    # How it works : Multiplies monthly salary self.salary * 12
    def calculate_annual_salary(self):
        return self.salary * 12

    # What is used : Instance method display_profile()
    def display_profile(self):
        print(f"ID: {self.emp_id:<6} | Name: {self.name:<15} | Dept: {self.department:<12} | Monthly: Rs.{self.salary:,.2f} | Annual: Rs.{self.calculate_annual_salary():,.2f}")

def main():
    print("=== EMPLOYEE DIRECTORY ===")
    emp1 = Employee("E101", "Suraj Sawant", 85000, "Engineering")
    emp2 = Employee("E102", "Rahul Sharma", 65000, "Marketing")

    emp1.display_profile()
    emp2.display_profile()

if __name__ == "__main__":
    main()
