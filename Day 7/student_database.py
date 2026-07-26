# Mini Project: Student Database System
# Features: Add Student, Delete Student, Update Student, Search Student, Display All Students

students = {
    101: {"name": "Suraj", "marks": 91, "grade": "A+"},
    102: {"name": "Rahul", "marks": 84, "grade": "A"}
}

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

def add_student():
    try:
        roll = int(input("Enter Roll Number (ID): "))
        if roll in students:
            print("Student ID already exists!")
            return
        name = input("Enter Student Name: ").strip()
        marks = float(input("Enter Marks: "))
        
        if marks >= 90:
            grade = "A+"
        elif marks >= 80:
            grade = "A"
        elif marks >= 70:
            grade = "B"
        elif marks >= 50:
            grade = "C"
        else:
            grade = "F"

        students[roll] = {"name": name, "marks": marks, "grade": grade}
        print(f"Student '{name}' added successfully!")
    except ValueError:
        print("Invalid input! Roll number and marks must be numeric.")

def delete_student():
    try:
        roll = int(input("Enter Roll Number to delete: "))
        if roll in students:
            removed = students.pop(roll)
            print(f"Student '{removed['name']}' (ID: {roll}) removed successfully!")
        else:
            print("Student ID not found.")
    except ValueError:
        print("Invalid Roll Number.")

def update_student():
    try:
        roll = int(input("Enter Roll Number to update: "))
        if roll in students:
            print(f"Current details for ID {roll}: {students[roll]}")
            name = input("Enter new name (leave blank to keep current): ").strip()
            marks_input = input("Enter new marks (leave blank to keep current): ").strip()

            if name:
                students[roll]["name"] = name
            if marks_input:
                marks = float(marks_input)
                students[roll]["marks"] = marks
                students[roll]["grade"] = "A+" if marks >= 90 else ("A" if marks >= 80 else ("B" if marks >= 70 else "F"))
            print(f"Student ID {roll} updated successfully!")
        else:
            print("Student ID not found.")
    except ValueError:
        print("Invalid numeric value entered.")

def search_student():
    try:
        roll = int(input("Enter Roll Number to search: "))
        if roll in students:
            s = students[roll]
            print(f"\nID: {roll} | Name: {s['name']} | Marks: {s['marks']} | Grade: {s['grade']}")
        else:
            print("Student ID not found.")
    except ValueError:
        print("Invalid Roll Number.")

def display_all():
    if not students:
        print("No student records available.")
    else:
        print("\n---------------- ALL STUDENTS ----------------")
        print(f"{'ID':<6} {'Name':<15} {'Marks':<8} {'Grade':<6}")
        print("----------------------------------------------")
        for roll, info in students.items():
            print(f"{roll:<6} {info['name']:<15} {info['marks']:<8} {info['grade']:<6}")
        print("----------------------------------------------")

def main():
    while True:
        display_menu()
        choice = input("Enter choice (1-6): ").strip()
        if choice == "1":
            add_student()
        elif choice == "2":
            delete_student()
        elif choice == "3":
            update_student()
        elif choice == "4":
            search_student()
        elif choice == "5":
            display_all()
        elif choice == "6":
            print("Exiting Student Database System. Goodbye!")
            break
        else:
            print("Invalid choice! Enter 1-6.")

if __name__ == "__main__":
    main()
