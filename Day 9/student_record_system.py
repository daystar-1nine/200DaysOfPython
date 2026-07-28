# ==============================================================================
# Program    : Mini Project: Student Record File System
# Objective  : Practice and master mini project: student record file system logic.
# Concept    : Add Student, View Students, Search Student, Exit (Stored in students.txt)
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

import os

filename = "students.txt"


# What is used : Function definition 'def add_student'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def add_student():
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
    roll = input("Enter Roll Number: ").strip()
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
    name = input("Enter Student Name: ").strip()
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
    marks = input("Enter Marks: ").strip()

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if not roll or not name or not marks:
        print("All fields are required!")
        return

    record = f"{roll},{name},{marks}\n"

# What is used : Context manager 'with open(...)'
# Why it is used: Guarantees file stream handles are automatically closed after execution
# How it works : Calls __enter__ to open stream and __exit__ to close stream safely
    with open(filename, "a", encoding="utf-8") as f:
        f.write(record)
    print(f"Student '{name}' record added successfully!")


# What is used : Function definition 'def view_students'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def view_students():
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        print("No student records found.")
        return

    print("\n---------------- STUDENT RECORDS ----------------")
    print(f"{'Roll No':<10} {'Name':<20} {'Marks':<10}")
    print("-------------------------------------------------")

# What is used : Context manager 'with open(...)'
# Why it is used: Guarantees file stream handles are automatically closed after execution
# How it works : Calls __enter__ to open stream and __exit__ to close stream safely
    with open(filename, "r", encoding="utf-8") as f:

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
        for line in f:
            parts = line.strip().split(",")
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
            if len(parts) == 3:
                roll, name, marks = parts
                print(f"{roll:<10} {name:<20} {marks:<10}")
    print("-------------------------------------------------")


# What is used : Function definition 'def search_student'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def search_student():
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        print("No student records found.")
        return

# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
    search_roll = input("Enter Roll Number to search: ").strip()
    found = False


# What is used : Context manager 'with open(...)'
# Why it is used: Guarantees file stream handles are automatically closed after execution
# How it works : Calls __enter__ to open stream and __exit__ to close stream safely
    with open(filename, "r", encoding="utf-8") as f:

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
        for line in f:
            parts = line.strip().split(",")
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
            if len(parts) == 3 and parts[0] == search_roll:
                print(f"\nRecord Found -> Roll No: {parts[0]} | Name: {parts[1]} | Marks: {parts[2]}")
                found = True
                break

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if not found:
        print(f"Student with Roll Number '{search_roll}' not found.")


# What is used : Function definition 'def main'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def main():

# What is used : while loop condition
# Why it is used: Continuously executes code block as long as condition evaluates to True
    while True:
        print("\n=== Student Record File System ===")
        print("1. Add Student")
        print("2. View Students")
        print("3. Search Student")
        print("4. Exit")

# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
        choice = input("Select choice (1-4): ").strip()
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
        if choice == "1":
            add_student()
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "2":
            view_students()
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "3":
            search_student()
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "4":
            print("Exiting Student Record System. Goodbye!")
            break
        else:
            print("Invalid choice! Enter 1-4.")

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if __name__ == "__main__":
    main()
