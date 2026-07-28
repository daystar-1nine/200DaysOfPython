# ==============================================================================
# Program    : Challenge Project: Personal Diary
# Objective  : Practice and master challenge project: personal diary logic.
# Concept    : Write Entry, View Diary, Exit (Stored in diary.txt with Date/Time)
# Why Used   : Executes continuously as long as the specified boolean condition remains True. Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle.
# ==============================================================================

import os
from datetime import datetime

filename = "diary.txt"

def write_entry():
    message = input("Enter your diary entry:\n").strip()
    if not message:
        print("Diary entry cannot be empty!")
        return

    now = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    entry = f"[{now}]\n{message}\n----------------------------------------\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(entry)
    print("Diary entry saved successfully!")

def view_diary():
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        print("Your diary is currently empty.")
        return

    print("\n================ MY DIARY ================")
    with open(filename, "r", encoding="utf-8") as f:
        print(f.read())

def main():
    while True:
        print("\n--- Personal Diary ---")
        print("1. Write Entry")
        print("2. View Diary")
        print("3. Exit")

        choice = input("Select option (1-3): ").strip()
        if choice == "1":
            write_entry()
        elif choice == "2":
            view_diary()
        elif choice == "3":
            print("Exiting Personal Diary. Stay inspired!")
            break
        else:
            print("Invalid choice! Enter 1-3.")

if __name__ == "__main__":
    main()
