# ==============================================================================
# Program    : Challenge Project: Personal Diary
# Objective  : Practice and master challenge project: personal diary logic.
# Concept    : Write Entry, View Diary, Exit (Stored in diary.txt with Date/Time)
# Why Used   : Executes continuously as long as the specified boolean condition remains True. Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle.
# ==============================================================================

import os
from datetime import datetime

filename = "diary.txt"


# What is used : Function definition 'def write_entry'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def write_entry():
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
    message = input("Enter your diary entry:\n").strip()
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if not message:
        print("Diary entry cannot be empty!")
        return

    now = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    entry = f"[{now}]\n{message}\n----------------------------------------\n"


# What is used : Context manager 'with open(...)'
# Why it is used: Guarantees file stream handles are automatically closed after execution
# How it works : Calls __enter__ to open stream and __exit__ to close stream safely
    with open(filename, "a", encoding="utf-8") as f:
        f.write(entry)
    print("Diary entry saved successfully!")


# What is used : Function definition 'def view_diary'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def view_diary():
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        print("Your diary is currently empty.")
        return

    print("\n================ MY DIARY ================")

# What is used : Context manager 'with open(...)'
# Why it is used: Guarantees file stream handles are automatically closed after execution
# How it works : Calls __enter__ to open stream and __exit__ to close stream safely
    with open(filename, "r", encoding="utf-8") as f:
        print(f.read())


# What is used : Function definition 'def main'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def main():

# What is used : while loop condition
# Why it is used: Continuously executes code block as long as condition evaluates to True
    while True:
        print("\n--- Personal Diary ---")
        print("1. Write Entry")
        print("2. View Diary")
        print("3. Exit")

# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
        choice = input("Select option (1-3): ").strip()
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
        if choice == "1":
            write_entry()
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "2":
            view_diary()
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "3":
            print("Exiting Personal Diary. Stay inspired!")
            break
        else:
            print("Invalid choice! Enter 1-3.")

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if __name__ == "__main__":
    main()
