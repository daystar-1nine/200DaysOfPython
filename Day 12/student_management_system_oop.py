# ==============================================================================
# Program    : Student Management System (OOP List Container)
# Objective  : Manage multiple student objects in a list with grade calculation features.
# Concept    : Class Objects List Aggregation & Domain Logic Methods
# Why Used   : Encapsulates individual student records into objects and manages them in a Python list.
# ==============================================================================

# What is used : Class definition 'class Student:'
# Why it is used: Represents individual student academic data and grading methods
class Student:
    """Class representing individual student academic records."""

    def __init__(self, roll_no, name, marks_list):
        self.roll_no = roll_no
        self.name = name
        self.marks_list = marks_list  # List of marks (e.g. [80, 85, 90])

    # What is used : Instance method 'calculate_percentage(self)'
    # How it works : Sums marks_list elements and divides by maximum total marks (len * 100)
    def calculate_percentage(self):
        if not self.marks_list:
            return 0.0
        return (sum(self.marks_list) / (len(self.marks_list) * 100)) * 100

    # What is used : Instance method 'assign_grade(self)'
    # How it works : Evaluates percentage boundary and returns letter grade string
    def assign_grade(self):
        pct = self.calculate_percentage()
        if pct >= 90:
            return "A+"
        elif pct >= 80:
            return "A"
        elif pct >= 70:
            return "B"
        elif pct >= 60:
            return "C"
        elif pct >= 40:
            return "D"
        else:
            return "F"

    # What is used : Instance method 'display_record(self)'
    # Why it is used: Formats student record details into a single directory row
    def display_record(self):
        pct = self.calculate_percentage()
        grade = self.assign_grade()
        total_obtained = sum(self.marks_list)
        total_max = len(self.marks_list) * 100
        print(f"Roll: {self.roll_no:<6} | Name: {self.name:<15} | Marks: {total_obtained}/{total_max} | Percentage: {pct:.2f}% | Grade: {grade}")

# What is used : Container class 'class StudentManagementSystem:'
# Why it is used: Manages collection of Student objects inside a list
class StudentManagementSystem:
    """System container managing a list of Student objects."""

    def __init__(self):
        # What is used : Empty list attribute 'self.students' to store Student objects
        self.students = []

    # What is used : Method 'add_student(self, student_obj)'
    # How it works : Appends Student instance to self.students list
    def add_student(self, student_obj):
        self.students.append(student_obj)
        print(f"Student '{student_obj.name}' added successfully!")

    # What is used : Method 'display_all(self)'
    # How it works : Iterates through self.students list and calls display_record() on each object
    def display_all(self):
        if not self.students:
            print("No student records found in system.")
            return

        print("\n==========================================================================")
        print("                        STUDENT MANAGEMENT DIRECTORY                       ")
        print("==========================================================================")
        for std in self.students:
            std.display_record()
        print("==========================================================================\n")

def main():
    sms = StudentManagementSystem()

    # Pre-populating sample Student objects
    s1 = Student(101, "Suraj Sawant", [90, 88, 95])
    s2 = Student(102, "Rahul Sharma", [75, 80, 78])
    s3 = Student(103, "Amit Patel", [60, 65, 58])

    sms.add_student(s1)
    sms.add_student(s2)
    sms.add_student(s3)

    sms.display_all()

if __name__ == "__main__":
    main()
