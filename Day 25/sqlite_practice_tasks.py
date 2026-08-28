# ==============================================================================
# Program    : SQLite Practice Tasks (Tasks 1 to 10)
# Objective  : Demonstrate complete SQLite table creation, CRUD operations, and aggregate queries.
# Concept    : sqlite3 Module, Parameterized Queries & SQL Aggregates
# Why Used   : Complete walkthrough covering all 10 practice tasks in Day 25 requirements.
# ==============================================================================

import os
import sqlite3

DB_FILE = "practice_students.db"

def init_db(conn: sqlite3.Connection) -> None:
    """Task 1: Create students.db and students table."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            marks REAL
        )
    """)
    conn.commit()

def insert_10_students(conn: sqlite3.Connection) -> None:
    """Task 2: Insert 10 students using parameterized queries."""
    sample_students = [
        ("Suraj Sawant", 20, 88.5),
        ("Rahul Sharma", 21, 92.0),
        ("Priya Patel", 19, 84.0),
        ("Amit Kumar", 22, 76.5),
        ("Neha Verma", 20, 95.0),
        ("Vikram Singh", 23, 68.0),
        ("Ananya Das", 21, 81.5),
        ("Rohan Mehta", 20, 74.0),
        ("Sneha Gupta", 22, 89.0),
        ("Karan Malhotra", 19, 91.5)
    ]
    cursor = conn.cursor()
    # What is used : executemany with ? placeholders
    # Why it is used: Safely inserts list of tuples in single transaction safely
    cursor.executemany(
        "INSERT INTO students (name, age, marks) VALUES (?, ?, ?)",
        sample_students
    )
    conn.commit()
    print("[Task 2] Inserted 10 student records successfully.")

def display_all_students(conn: sqlite3.Connection) -> None:
    """Task 3: Display all students."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students")
    records = cursor.fetchall()
    print("\n--- [Task 3] ALL STUDENTS RECORD ---")
    for row in records:
        print(f"ID: {row[0]:<2} | Name: {row[1]:<16} | Age: {row[2]} | Marks: {row[3]}")

def find_students_above_80(conn: sqlite3.Connection) -> None:
    """Task 4: Find students with marks > 80."""
    cursor = conn.cursor()
    cursor.execute("SELECT name, marks FROM students WHERE marks > ? ORDER BY marks DESC", (80.0,))
    records = cursor.fetchall()
    print("\n--- [Task 4] STUDENTS WITH MARKS > 80 ---")
    for name, marks in records:
        print(f"Name: {name:<16} | Marks: {marks}")

def sort_students_by_marks(conn: sqlite3.Connection) -> None:
    """Task 5: Sort students by marks descending."""
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, marks FROM students ORDER BY marks DESC")
    records = cursor.fetchall()
    print("\n--- [Task 5] STUDENTS SORTED BY MARKS (DESCENDING) ---")
    for row in records:
        print(f"ID: {row[0]:<2} | Name: {row[1]:<16} | Marks: {row[2]}")

def update_student_marks(conn: sqlite3.Connection, student_id: int, new_marks: float) -> None:
    """Task 6: Update one student's marks."""
    cursor = conn.cursor()
    cursor.execute("UPDATE students SET marks = ? WHERE id = ?", (new_marks, student_id))
    conn.commit()
    print(f"\n[Task 6] Updated Student ID {student_id} marks to {new_marks}.")

def delete_student(conn: sqlite3.Connection, student_id: int) -> None:
    """Task 7: Delete one student."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
    conn.commit()
    print(f"\n[Task 7] Deleted Student ID {student_id}.")

def show_aggregates(conn: sqlite3.Connection) -> None:
    """Tasks 8, 9, 10: Count, Max, and Average Marks."""
    cursor = conn.cursor()
    
    # Task 8: Count
    cursor.execute("SELECT COUNT(*) FROM students")
    count = cursor.fetchone()[0]

    # Task 9: Max
    cursor.execute("SELECT MAX(marks) FROM students")
    max_marks = cursor.fetchone()[0]

    # Task 10: Average
    cursor.execute("SELECT AVG(marks) FROM students")
    avg_marks = cursor.fetchone()[0]

    print("\n--- [Tasks 8, 9, 10] AGGREGATE METRICS ---")
    print(f"Task 8  - Total Students Count : {count}")
    print(f"Task 9  - Highest Marks Found  : {max_marks}")
    print(f"Task 10 - Average Marks        : {avg_marks:.2f}")

def main() -> None:
    print("=== DAY 25: SQLITE PRACTICE TASKS 1 TO 10 ===")
    
    conn = sqlite3.connect(DB_FILE)
    try:
        init_db(conn)
        
        # Clear previous data for clean test run
        conn.cursor().execute("DELETE FROM students")
        conn.commit()

        insert_10_students(conn)
        display_all_students(conn)
        find_students_above_80(conn)
        sort_students_by_marks(conn)
        
        # Task 6: Update student 1 marks to 98.0
        update_student_marks(conn, 1, 98.0)
        
        # Task 7: Delete student 6 (Vikram Singh)
        delete_student(conn, 6)
        
        show_aggregates(conn)
    finally:
        conn.close()

    # Cleanup test db
    if os.path.exists(DB_FILE):
        try:
            os.remove(DB_FILE)
        except OSError:
            pass

if __name__ == "__main__":
    main()
