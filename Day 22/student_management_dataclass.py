# ==============================================================================
# Program    : Student Management System (Mini Project)
# Objective  : Manage student data using dataclasses with methods for average, percentage, grade, and display.
# Concept    : Dataclass Methods & Formatted Terminal Output
# Why Used   : Encapsulates student evaluation logic cleanly using type annotations.
# ==============================================================================

from dataclasses import dataclass, field

@dataclass
class Student:
    id: int
    name: str
    marks: list[int] = field(default_factory=list)

    def average(self) -> float:
        if not self.marks:
            return 0.0
        return sum(self.marks) / len(self.marks)

    def percentage(self) -> float:
        """Assumes each subject mark is out of 100."""
        return self.average()

    def grade(self) -> str:
        avg: float = self.average()
        if avg >= 90:
            return "A+"
        elif avg >= 80:
            return "A"
        elif avg >= 70:
            return "B"
        elif avg >= 60:
            return "C"
        elif avg >= 40:
            return "D"
        else:
            return "F"

    def display(self) -> None:
        print("\n------------------ STUDENT RECORD ------------------")
        print(f"ID         : {self.id}")
        print(f"Name       : {self.name}")
        print(f"Marks      : {self.marks}")
        print(f"Average    : {self.average():.2f}")
        print(f"Percentage : {self.percentage():.2f}%")
        print(f"Grade      : {self.grade()}")
        print("----------------------------------------------------\n")

def main() -> None:
    print("==========================================================")
    print("              STUDENT MANAGEMENT SYSTEM                   ")
    print("==========================================================")

    s1 = Student(id=101, name="Suraj Sawant", marks=[85, 91, 88, 92, 80])
    s2 = Student(id=102, name="Priya Sharma", marks=[95, 98, 92, 96, 94])

    s1.display()
    s2.display()

if __name__ == "__main__":
    main()
