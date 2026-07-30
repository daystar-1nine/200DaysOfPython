# ==============================================================================
# Program    : Employee Management System (Advanced OOP)
# Objective  : Model organization staff with inheritance and method overriding.
# Concept    : Inheritance, Method Overriding, `super()` & Polymorphism
# Why Used   : Base Employee class provides common profile; Manager and Developer override salary calculation.
# ==============================================================================

# What is used : Parent Base Class 'class Employee:'
class Employee:
    """Base Employee class."""

    def __init__(self, emp_id, name, base_salary, department):
        self.emp_id = emp_id
        self.name = name
        self.base_salary = base_salary
        self.department = department

    # What is used : Base method calculate_salary()
    def calculate_salary(self):
        return self.base_salary

    # What is used : Base method display_details()
    def display_details(self):
        print(f"ID: {self.emp_id:<6} | Name: {self.name:<15} | Dept: {self.department:<12} | Total Salary: Rs.{self.calculate_salary():,.2f}")

# What is used : Subclass 'class Manager(Employee):'
class Manager(Employee):
    """Manager subclass with bonus compensation."""

    def __init__(self, emp_id, name, base_salary, department, bonus):
        # Delegate attribute initialization to Employee parent constructor
        super().__init__(emp_id, name, base_salary, department)
        self.bonus = bonus

    # What is used : Method Overriding of calculate_salary()
    # How it works : Adds bonus to base salary for managers
    def calculate_salary(self):
        return self.base_salary + self.bonus

    # What is used : Method Overriding of display_details()
    def display_details(self):
        print(f"ID: {self.emp_id:<6} | Name: {self.name:<15} (Manager)   | Dept: {self.department:<12} | Total Salary: Rs.{self.calculate_salary():,.2f} (Bonus: Rs.{self.bonus:,.2f})")

# What is used : Subclass 'class Developer(Employee):'
class Developer(Employee):
    """Developer subclass with overtime compensation."""

    def __init__(self, emp_id, name, base_salary, department, overtime_hours, hourly_rate=500):
        super().__init__(emp_id, name, base_salary, department)
        self.overtime_hours = overtime_hours
        self.hourly_rate = hourly_rate

    # What is used : Method Overriding of calculate_salary()
    def calculate_salary(self):
        overtime_pay = self.overtime_hours * self.hourly_rate
        return self.base_salary + overtime_pay

    def display_details(self):
        overtime_pay = self.overtime_hours * self.hourly_rate
        print(f"ID: {self.emp_id:<6} | Name: {self.name:<15} (Developer) | Dept: {self.department:<12} | Total Salary: Rs.{self.calculate_salary():,.2f} (OT Pay: Rs.{overtime_pay:,.2f})")

def main():
    print("=========================================================================================")
    print("                             EMPLOYEE MANAGEMENT SYSTEM                                ")
    print("=========================================================================================")

    # What is used : Heterogeneous polymorphic list of Employee, Manager, and Developer objects
    staff = [
        Employee("E101", "Amit Patel", 50000, "Support"),
        Manager("M201", "Suraj Sawant", 90000, "Engineering", bonus=25000),
        Developer("D301", "Rahul Sharma", 70000, "Engineering", overtime_hours=20)
    ]

    # Polymorphic method invocation display_details()
    for emp in staff:
        emp.display_details()
    print("=========================================================================================\n")

if __name__ == "__main__":
    main()
