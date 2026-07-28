# ==============================================================================
# Program    : Bonus Challenge: Expense Tracker
# Objective  : Practice and master bonus challenge: expense tracker logic.
# Concept    : Add Expense, View Expenses, Calculate Total Expense (Stored in expenses.txt)
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

import os

filename = "expenses.txt"


# What is used : Function definition 'def add_expense'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def add_expense():
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
    category = input("Enter Expense Category (e.g. Food, Travel, Books): ").strip()
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
    amount_str = input("Enter Amount (Rs.): ").strip()

    try:
        amount = float(amount_str)
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
        if amount <= 0:
            print("Amount must be greater than zero.")
            return

        record = f"{category},{amount}\n"

# What is used : Context manager 'with open(...)'
# Why it is used: Guarantees file stream handles are automatically closed after execution
# How it works : Calls __enter__ to open stream and __exit__ to close stream safely
        with open(filename, "a", encoding="utf-8") as f:
            f.write(record)
        print(f"Expense '{category}: Rs.{amount}' added successfully!")
    except ValueError:
        print("Invalid amount! Please enter a numerical value.")


# What is used : Function definition 'def view_expenses'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def view_expenses():
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        print("No expenses recorded yet.")
        return

    print("\n-------------- EXPENSES LIST --------------")
    print(f"{'Category':<20} {'Amount (Rs.)':<15}")
    print("-------------------------------------------")

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
            if len(parts) == 2:
                print(f"{parts[0]:<20} Rs.{float(parts[1]):<15.2f}")
    print("-------------------------------------------")


# What is used : Function definition 'def calculate_total'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def calculate_total():
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        print("Total Expenses: Rs.0.00")
        return

    total = 0.0

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
            if len(parts) == 2:
                try:
                    total += float(parts[1])
                except ValueError:
                    pass

    print(f"\nGRAND TOTAL EXPENSE: Rs.{total:.2f}")


# What is used : Function definition 'def main'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def main():

# What is used : while loop condition
# Why it is used: Continuously executes code block as long as condition evaluates to True
    while True:
        print("\n--- Expense Tracker ---")
        print("1. Add Expense")
        print("2. View Expenses")
        print("3. Calculate Total Expense")
        print("4. Exit")

# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
        choice = input("Select option (1-4): ").strip()
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
        if choice == "1":
            add_expense()
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "2":
            view_expenses()
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "3":
            calculate_total()
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "4":
            print("Exiting Expense Tracker. Goodbye!")
            break
        else:
            print("Invalid choice! Enter 1-4.")

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if __name__ == "__main__":
    main()
