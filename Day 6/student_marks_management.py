# ==============================================================================
# Program    : Mini Project: Student Marks Management System
# Objective  : Practice and master mini project: student marks management system logic.
# Concept    : Add student marks, Display all marks, Calculate average, Highest & Lowest mark
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================


# What is used : Function definition 'def display_menu'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def display_menu():
    print("\n==================================")
    print("  Student Marks Management System")
    print("==================================")
    print("1. Add Student Mark")
    print("2. Display All Marks")
    print("3. View Analytics (Highest, Lowest, Average)")
    print("4. Exit")


# What is used : Function definition 'def main'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def main():
    marks = [78, 89, 91, 67, 85]  # Pre-populated initial sample marks


# What is used : while loop condition
# Why it is used: Continuously executes code block as long as condition evaluates to True
    while True:
        display_menu()
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
        choice = input("Enter your choice (1-4): ").strip()

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
        if choice == "1":
            try:
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
                mark = float(input("Enter student mark: "))
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
                if 0 <= mark <= 100:
                    marks.append(mark)
                    print(f"Mark {mark} added successfully!")
                else:
                    print("Please enter a valid mark between 0 and 100.")
            except ValueError:
                print("Invalid input! Please enter a numerical value.")

# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "2":
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
            if not marks:
                print("No marks recorded yet.")
            else:
                print("\n------ Student Marks ------")

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
                for idx, m in enumerate(marks, start=1):
                    print(f"Student {idx}: {m}")

# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "3":
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
            if not marks:
                print("No marks recorded yet.")
            else:
                highest = max(marks)
                lowest = min(marks)
                average = sum(marks) / len(marks)

                print("\n------ Analytics ------")
                print(f"Total Students : {len(marks)}")
                print(f"Highest Mark   : {highest}")
                print(f"Lowest Mark    : {lowest}")
                print(f"Average Mark   : {average:.2f}")

# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "4":
            print("Exiting Student Marks Management System. Goodbye!")
            break
        else:
            print("Invalid choice! Please select 1, 2, 3, or 4.")

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if __name__ == "__main__":
    main()
