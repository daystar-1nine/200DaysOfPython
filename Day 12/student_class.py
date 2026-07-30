# ==============================================================================
# Program    : Student Class Implementation
# Objective  : Model student entities using Object-Oriented Programming (OOP).
# Concept    : Classes, Objects, Constructor (__init__), Attributes & Instance Methods
# Why Used   : Demonstrates binding state (name, roll_no, marks) and behavior into a Student class.
# ==============================================================================

# What is used : Class definition 'class Student:'
# Why it is used: Defines the template/blueprint for student objects
# How it works : Allocates a class type in Python namespace
class Student:
    """Class representing a student entity."""

    # What is used : Constructor method '__init__(self, name, roll_no, marks)'
    # Why it is used: Automatically initializes instance attributes upon object creation
    # How it works : Self points to newly created instance; binds parameter values to instance variables
    def __init__(self, name, roll_no, marks):
        # What is used : Instance attribute assignments
        self.name = name
        self.roll_no = roll_no
        self.marks = marks

    # What is used : Instance method 'display_info(self)'
    # Why it is used: Prints formatted student details by accessing instance attributes via self
    # How it works : Accesses self.name, self.roll_no, self.marks for the calling object
    def display_info(self):
        print(f"Student Name : {self.name}")
        print(f"Roll Number  : {self.roll_no}")
        print(f"Marks        : {self.marks} / 100\n")

def main():
    print("=== Student Objects Demonstration ===")

    # What is used : Object Instantiation 'Student("Suraj", 101, 95)'
    # Why it is used: Creates concrete student object in memory and invokes __init__()
    # How it works : Allocates memory, binds 'Suraj', 101, 95 to self, returns reference to student1
    student1 = Student("Suraj", 101, 95)
    student2 = Student("Rahul", 102, 88)

    # What is used : Instance method calls 'student1.display_info()'
    # How it works : Passes student1 into self parameter of display_info()
    student1.display_info()
    student2.display_info()

if __name__ == "__main__":
    main()
