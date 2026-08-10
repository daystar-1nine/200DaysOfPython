# ==============================================================================
# Program    : Student TypedDict Schema (Task 3)
# Objective  : Define fixed dictionary key structure using typing.TypedDict.
# Concept    : Structural Dictionary Typing (TypedDict)
# Why Used   : Provides type safety and field documentation for dictionary data payloads.
# ==============================================================================

from typing import TypedDict

# What is used : TypedDict subclass definition
# Why it is used: Specifies mandatory keys and value data types for student dictionaries
class StudentDict(TypedDict):
    name: str
    age: int
    cgpa: float
    skills: list[str]

def format_student(student: StudentDict) -> str:
    """Formats student dictionary record into readable string."""
    skills_str: str = ", ".join(student["skills"])
    return f"Student: {student['name']} (Age: {student['age']}) | CGPA: {student['cgpa']} | Skills: {skills_str}"

def main() -> None:
    print("=== TASK 3: STUDENT TYPEDDICT DEMO ===")
    
    # Instantiate student dictionary adhering to StudentDict schema
    student1: StudentDict = {
        "name": "Suraj Sawant",
        "age": 20,
        "cgpa": 8.85,
        "skills": ["Python", "Machine Learning", "FastAPI"]
    }

    print(format_student(student1))

if __name__ == "__main__":
    main()
