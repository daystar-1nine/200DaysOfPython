# ==============================================================================
# Program    : Bonus Challenge: Expense Tracker
# Objective  : Practice and master bonus challenge: expense tracker logic.
# Concept    : Add Expense, View Expenses, Calculate Total Expense (Stored in expenses.txt)
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

import os

filename = "expenses.txt"

def add_expense():
    category = input("Enter Expense Category (e.g. Food, Travel, Books): ").strip()
    amount_str = input("Enter Amount (Rs.): ").strip()

    try:
        amount = float(amount_str)
        if amount <= 0:
            print("Amount must be greater than zero.")
            return

        record = f"{category},{amount}\n"
        with open(filename, "a", encoding="utf-8") as f:
            f.write(record)
        print(f"Expense '{category}: Rs.{amount}' added successfully!")
    except ValueError:
        print("Invalid amount! Please enter a numerical value.")

def view_expenses():
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        print("No expenses recorded yet.")
        return

    print("\n-------------- EXPENSES LIST --------------")
    print(f"{'Category':<20} {'Amount (Rs.)':<15}")
    print("-------------------------------------------")
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2:
                print(f"{parts[0]:<20} Rs.{float(parts[1]):<15.2f}")
    print("-------------------------------------------")

def calculate_total():
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        print("Total Expenses: Rs.0.00")
        return

    total = 0.0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2:
                try:
                    total += float(parts[1])
                except ValueError:
                    pass

    print(f"\nGRAND TOTAL EXPENSE: Rs.{total:.2f}")

def main():
    while True:
        print("\n--- Expense Tracker ---")
        print("1. Add Expense")
        print("2. View Expenses")
        print("3. Calculate Total Expense")
        print("4. Exit")

        choice = input("Select option (1-4): ").strip()
        if choice == "1":
            add_expense()
        elif choice == "2":
            view_expenses()
        elif choice == "3":
            calculate_total()
        elif choice == "4":
            print("Exiting Expense Tracker. Goodbye!")
            break
        else:
            print("Invalid choice! Enter 1-4.")

if __name__ == "__main__":
    main()
