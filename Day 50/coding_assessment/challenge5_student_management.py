"""
===============================================================================
DAY 50 — CODING CHALLENGE 5: INTERACTIVE STUDENT MANAGEMENT APPLICATION
===============================================================================
This module provides a complete CLI application supporting:
1. Add Student
2. View Students
3. Search Student
4. Update Student
5. Delete Student
6. Calculate Average
7. Exit
Uses functions, dictionaries, loops, and robust exception handling.
===============================================================================
"""

from typing import Dict, Any

class StudentManagerApp:
    """Student management application domain logic."""

    def __init__(self) -> None:
        # What is used: Dictionary collection storing student data.
        # Why it is used: Provides O(1) CRUD access indexed by student ID.
        # How it works: Maps student_id to dict payload containing name and marks.
        self.students: Dict[str, Dict[str, Any]] = {}

    def add_student(self, student_id: str, name: str, marks: float) -> str:
        """Add a student record."""
        if student_id in self.students:
            raise ValueError(f"Student ID '{student_id}' already exists.")
        if marks < 0 or marks > 100:
            raise ValueError("Marks must be between 0 and 100.")
        self.students[student_id] = {"name": name, "marks": marks}
        return f"Student '{name}' added successfully."

    def view_students(self) -> str:
        """View all registered students."""
        if not self.students:
            return "No student records found."
        lines = ["=== STUDENT LIST ==="]
        for s_id, data in self.students.items():
            lines.append(f"ID: {s_id} | Name: {data['name']} | Marks: {data['marks']:.2f}")
        return "\n".join(lines)

    def search_student(self, student_id: str) -> str:
        """Search student by ID."""
        if student_id not in self.students:
            raise KeyError(f"Student ID '{student_id}' not found.")
        data = self.students[student_id]
        return f"Found -> ID: {student_id} | Name: {data['name']} | Marks: {data['marks']:.2f}"

    def update_student(self, student_id: str, name: str | None = None, marks: float | None = None) -> str:
        """Update student details."""
        if student_id not in self.students:
            raise KeyError(f"Student ID '{student_id}' not found.")
        if name:
            self.students[student_id]["name"] = name
        if marks is not None:
            if marks < 0 or marks > 100:
                raise ValueError("Marks must be between 0 and 100.")
            self.students[student_id]["marks"] = marks
        return f"Student ID '{student_id}' updated successfully."

    def delete_student(self, student_id: str) -> str:
        """Delete student record by ID."""
        if student_id not in self.students:
            raise KeyError(f"Student ID '{student_id}' not found.")
        del self.students[student_id]
        return f"Student ID '{student_id}' deleted successfully."

    def calculate_average(self) -> float:
        """Calculate average marks across all students."""
        if not self.students:
            return 0.0
        total = sum(s["marks"] for s in self.students.values())
        return total / len(self.students)


if __name__ == "__main__":
    app = StudentManagerApp()
    print("Testing Student Management CLI Logic...")
    print(app.add_student("101", "Suraj", 95.0))
    print(app.add_student("102", "Ananya", 88.0))
    print(app.view_students())
    print(app.search_student("101"))
    print(app.update_student("102", marks=91.5))
    print(f"Average Marks: {app.calculate_average():.2f}")
    print(app.delete_student("101"))
    print(f"New Average Marks: {app.calculate_average():.2f}")
    print("✅ Challenge 5 Passed!")
