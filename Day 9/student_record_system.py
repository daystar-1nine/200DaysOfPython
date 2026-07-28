# ==============================================================================
# Program    : Mini Project: Student Record File System
# Objective  : Practice and master mini project: student record file system logic.
# Concept    : Add Student, View Students, Search Student, Exit (Stored in students.txt)
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

import os

filename = "students.txt"

def add_student():
    roll = input("Enter Roll Number: ").strip()
    name = input("Enter Student Name: ").strip()
    marks = input("Enter Marks: ").strip()

    if not roll or not name or not marks:
        print("All fields are required!")
        return

    record = f"{roll},{name},{marks}\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(record)
    print(f"Student '{name}' record added successfully!")

def view_students():
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        print("No student records found.")
        return

    print("\n---------------- STUDENT RECORDS ----------------")
    print(f"{'Roll No':<10} {'Name':<20} {'Marks':<10}")
    print("-------------------------------------------------")
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                roll, name, marks = parts
                print(f"{roll:<10} {name:<20} {marks:<10}")
    print("-------------------------------------------------")

def search_student():
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        print("No student records found.")
        return

    search_roll = input("Enter Roll Number to search: ").strip()
    found = False

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3 and parts[0] == search_roll:
                print(f"\nRecord Found -> Roll No: {parts[0]} | Name: {parts[1]} | Marks: {parts[2]}")
                found = True
                break

    if not found:
        print(f"Student with Roll Number '{search_roll}' not found.")

def main():
    while True:
        print("\n=== Student Record File System ===")
        print("1. Add Student")
        print("2. View Students")
        print("3. Search Student")
        print("4. Exit")

        choice = input("Select choice (1-4): ").strip()
        if choice == "1":
            add_student()
        elif choice == "2":
            view_students()
        elif choice == "3":
            search_student()
        elif choice == "4":
            print("Exiting Student Record System. Goodbye!")
            break
        else:
            print("Invalid choice! Enter 1-4.")

if __name__ == "__main__":
    main()
