import pytest
from student_management_system_oop import Student, StudentManagementSystem

def test_student_percentage_calculation():
    """Verify correct percentage logic for valid marks list."""
    student = Student(101, "Suraj", [90, 88, 95])
    assert student.calculate_percentage() == pytest.approx(91.0)

def test_student_empty_marks():
    """Verify empty marks list defaults to 0%."""
    student = Student(102, "No Marks", [])
    assert student.calculate_percentage() == 0.0

def test_student_assign_grade_logic():
    """Verify letter grade assignment bounds."""
    assert Student(1, "A+", [95, 95]).assign_grade() == "A+"
    assert Student(2, "A", [80, 85]).assign_grade() == "A"
    assert Student(3, "B", [70, 75]).assign_grade() == "B"
    assert Student(4, "F", [30, 35]).assign_grade() == "F"

def test_sms_add_student():
    """Verify StudentManagementSystem appends students to internal list."""
    sms = StudentManagementSystem()
    assert len(sms.students) == 0
    
    student = Student(101, "Suraj", [90, 90])
    sms.add_student(student)
    assert len(sms.students) == 1
    assert sms.students[0].name == "Suraj"
