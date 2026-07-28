# ==============================================================================
# Program    : Mini Project: Student Database System
# Objective  : Practice and master mini project: student database system logic.
# Concept    : Add Student, Delete Student, Update Student, Search Student, Display All Students
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

students = {
    101: {"name": "Suraj", "marks": 91, "grade": "A+"},
    102: {"name": "Rahul", "marks": 84, "grade": "A"}
}


# What is used : Function definition 'def display_menu'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def display_menu():
    print("\n===============================")
    print("    Student Database System")
    print("===============================")
    print("1. Add Student")
    print("2. Delete Student")
    print("3. Update Student")
    print("4. Search Student")
    print("5. Display All Students")
    print("6. Exit")


# What is used : Function definition 'def add_student'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def add_student():
    try:
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
        roll = int(input("Enter Roll Number (ID): "))
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
        if roll in students:
            print("Student ID already exists!")
            return
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
        name = input("Enter Student Name: ").strip()
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
        marks = float(input("Enter Marks: "))
        
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
        if marks >= 90:
            grade = "A+"
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif marks >= 80:
            grade = "A"
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif marks >= 70:
            grade = "B"
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif marks >= 50:
            grade = "C"
        else:
            grade = "F"

        students[roll] = {"name": name, "marks": marks, "grade": grade}
        print(f"Student '{name}' added successfully!")
    except ValueError:
        print("Invalid input! Roll number and marks must be numeric.")


# What is used : Function definition 'def delete_student'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def delete_student():
    try:
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
        roll = int(input("Enter Roll Number to delete: "))
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
        if roll in students:
            removed = students.pop(roll)
            print(f"Student '{removed['name']}' (ID: {roll}) removed successfully!")
        else:
            print("Student ID not found.")
    except ValueError:
        print("Invalid Roll Number.")


# What is used : Function definition 'def update_student'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def update_student():
    try:
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
        roll = int(input("Enter Roll Number to update: "))
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
        if roll in students:
            print(f"Current details for ID {roll}: {students[roll]}")
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
            name = input("Enter new name (leave blank to keep current): ").strip()
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
            marks_input = input("Enter new marks (leave blank to keep current): ").strip()

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
            if name:
                students[roll]["name"] = name
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
            if marks_input:
                marks = float(marks_input)
                students[roll]["marks"] = marks
                students[roll]["grade"] = "A+" if marks >= 90 else ("A" if marks >= 80 else ("B" if marks >= 70 else "F"))
            print(f"Student ID {roll} updated successfully!")
        else:
            print("Student ID not found.")
    except ValueError:
        print("Invalid numeric value entered.")


# What is used : Function definition 'def search_student'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def search_student():
    try:
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
        roll = int(input("Enter Roll Number to search: "))
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
        if roll in students:
            s = students[roll]
            print(f"\nID: {roll} | Name: {s['name']} | Marks: {s['marks']} | Grade: {s['grade']}")
        else:
            print("Student ID not found.")
    except ValueError:
        print("Invalid Roll Number.")


# What is used : Function definition 'def display_all'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def display_all():
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if not students:
        print("No student records available.")
    else:
        print("\n---------------- ALL STUDENTS ----------------")
        print(f"{'ID':<6} {'Name':<15} {'Marks':<8} {'Grade':<6}")
        print("----------------------------------------------")

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
        for roll, info in students.items():
            print(f"{roll:<6} {info['name']:<15} {info['marks']:<8} {info['grade']:<6}")
        print("----------------------------------------------")


# What is used : Function definition 'def main'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def main():

# What is used : while loop condition
# Why it is used: Continuously executes code block as long as condition evaluates to True
    while True:
        display_menu()
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
        choice = input("Enter choice (1-6): ").strip()
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
        if choice == "1":
            add_student()
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "2":
            delete_student()
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "3":
            update_student()
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "4":
            search_student()
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "5":
            display_all()
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "6":
            print("Exiting Student Database System. Goodbye!")
            break
        else:
            print("Invalid choice! Enter 1-6.")

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if __name__ == "__main__":
    main()
