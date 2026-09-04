"""
===============================================================================
DAY 52 — TEST SERVICES MODULE
===============================================================================
This test module verifies business logic operations including student CRUD,
auto ID generation, search filtering, and statistical aggregations.
===============================================================================
"""

import pytest
from app.models import Student
from app.services import (
    generate_next_id,
    add_student,
    update_student,
    delete_student,
    search_students,
    calculate_average,
    get_highest_scorer,
    get_lowest_scorer,
)


def test_generate_next_id(sample_students):
    """Verify unique incrementing ID generation logic."""
    # What is used: generate_next_id service function.
    # Why it is used: Ensures newly created student receives max(existing_id) + 1.
    # How it works: Passes sample_students (max ID = 5); expects next ID to be 6.
    assert generate_next_id(sample_students) == 6
    assert generate_next_id([]) == 1


def test_add_student(sample_students):
    """Verify adding a new student to list container."""
    # What is used: add_student function.
    # Why it is used: Validates adding new Student record and preventing duplicate IDs.
    # How it works: Adds Student(6,...); verifies total count increases to 6.
    new_student = Student(6, "Vikram", 22, "AI & Robotics", 88.0)
    added = add_student(sample_students, new_student)
    assert added.id == 6
    assert len(sample_students) == 6


def test_add_student_duplicate_id(sample_students):
    """Verify ValueError raised when attempting to add student with existing ID."""
    # What is used: pytest.raises with ValueError.
    # Why it is used: Enforces primary key ID uniqueness constraint in memory.
    # How it works: Attempts adding Student(1,...); asserts ValueError is raised.
    dup_student = Student(1, "Duplicate", 20, "CS", 80.0)
    with pytest.raises(ValueError, match="already exists"):
        add_student(sample_students, dup_student)


def test_update_student(sample_students):
    """Verify updating student attributes by ID."""
    # What is used: update_student function with keyword arguments.
    # Why it is used: Updates specific fields of an existing student object.
    # How it works: Updates ID 1 name and marks; asserts updated values match.
    updated = update_student(sample_students, 1, name="Rahul Sharma", marks=90.0)
    assert updated.name == "Rahul Sharma"
    assert updated.marks == 90.0


def test_update_student_not_found(sample_students):
    """Verify KeyError raised when updating non-existent student ID."""
    # What is used: pytest.raises with KeyError.
    # Why it is used: Handles invalid student lookup gracefully.
    # How it works: Attempts updating ID 999; asserts KeyError.
    with pytest.raises(KeyError, match="not found"):
        update_student(sample_students, 999, name="Unknown")


def test_delete_student(sample_students):
    """Verify deleting a student by ID."""
    # What is used: delete_student function.
    # Why it is used: Removes matching student object from list container.
    # How it works: Deletes ID 1; asserts deleted object ID is 1 and length is 4.
    deleted = delete_student(sample_students, 1)
    assert deleted.id == 1
    assert len(sample_students) == 4


def test_delete_student_not_found(sample_students):
    """Verify KeyError raised when deleting non-existent student ID."""
    # What is used: pytest.raises with KeyError.
    # Why it is used: Validates exception raising for invalid delete targets.
    # How it works: Attempts deleting ID 999; asserts KeyError.
    with pytest.raises(KeyError, match="not found"):
        delete_student(sample_students, 999)


def test_search_students(sample_students):
    """Verify multi-field search filtering by ID, name, or course."""
    # What is used: search_students function with query strings.
    # Why it is used: Filters students matching substring search terms.
    # How it works: Tests searching by course "Data Science" and name "Rahul".
    ds_results = search_students(sample_students, "Data Science")
    assert len(ds_results) == 2

    rahul_results = search_students(sample_students, "Rahul")
    assert len(rahul_results) == 1
    assert rahul_results[0].name == "Rahul"

    id_results = search_students(sample_students, "3")
    assert len(id_results) == 1
    assert id_results[0].id == 3


def test_calculate_average(sample_students):
    """Verify mean marks calculation logic."""
    # What is used: calculate_average function.
    # Why it is used: Computes overall academic average across all students.
    # How it works: Sum of marks (85+95+67+78+82) / 5 = 407 / 5 = 81.4.
    assert calculate_average(sample_students) == 81.4
    assert calculate_average([]) == 0.0


def test_get_highest_and_lowest_scorer(sample_students):
    """Verify highest and lowest scoring student identification."""
    # What is used: get_highest_scorer and get_lowest_scorer functions.
    # Why it is used: Identifies top performer and lowest performer in student roster.
    # How it works: Aisha has max 95.0, Rohan has min 67.0.
    highest = get_highest_scorer(sample_students)
    lowest = get_lowest_scorer(sample_students)
    assert highest.name == "Aisha"
    assert lowest.name == "Rohan"


def test_highest_lowest_empty_list():
    """Verify ValueError raised when computing min/max on empty list."""
    # What is used: pytest.raises with ValueError on empty lists.
    # Why it is used: Handles empty list edge case defensively.
    # How it works: Asserts ValueError when calling get_highest_scorer or get_lowest_scorer on [].
    with pytest.raises(ValueError, match="empty student list"):
        get_highest_scorer([])

    with pytest.raises(ValueError, match="empty student list"):
        get_lowest_scorer([])
