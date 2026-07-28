# ==============================================================================
# Program    : Mini Project: Student Marks Management System
# Objective  : Practice and master mini project: student marks management system logic.
# Concept    : Add student marks, Display all marks, Calculate average, Highest & Lowest mark
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

def display_menu():
    print("\n==================================")
    print("  Student Marks Management System")
    print("==================================")
    print("1. Add Student Mark")
    print("2. Display All Marks")
    print("3. View Analytics (Highest, Lowest, Average)")
    print("4. Exit")

def main():
    marks = [78, 89, 91, 67, 85]  # Pre-populated initial sample marks

    while True:
        display_menu()
        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            try:
                mark = float(input("Enter student mark: "))
                if 0 <= mark <= 100:
                    marks.append(mark)
                    print(f"Mark {mark} added successfully!")
                else:
                    print("Please enter a valid mark between 0 and 100.")
            except ValueError:
                print("Invalid input! Please enter a numerical value.")

        elif choice == "2":
            if not marks:
                print("No marks recorded yet.")
            else:
                print("\n------ Student Marks ------")
                for idx, m in enumerate(marks, start=1):
                    print(f"Student {idx}: {m}")

        elif choice == "3":
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

        elif choice == "4":
            print("Exiting Student Marks Management System. Goodbye!")
            break
        else:
            print("Invalid choice! Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
