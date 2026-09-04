"""
===============================================================================
DAY 51 — SERVICE LAYER UNIT TESTS
===============================================================================
This module tests calculate_average, highest/lowest scorer, count, filtering,
and performance distribution tallying.
===============================================================================
"""

import pytest
from typing import List
from app.models import Student
from app.enums import PerformanceLevel, get_performance_level
from app.services import (
    calculate_average,
    get_highest_scorer,
    get_lowest_scorer,
    count_students,
    filter_students_by_marks,
    get_performance_distribution,
)


def test_calculate_average_multiple_students(sample_students: List[Student]) -> None:
    """Test calculating mean average marks for multiple students."""
    # (85 + 92 + 67 + 78) / 4 = 322 / 4 = 80.5
    assert calculate_average(sample_students) == 80.5


def test_calculate_average_single_student(single_student: List[Student]) -> None:
    """Test calculating average marks for single student."""
    assert calculate_average(single_student) == 95.0


def test_calculate_average_empty_list() -> None:
    """Test calculating average for empty list returns 0.0."""
    assert calculate_average([]) == 0.0


def test_get_highest_scorer(sample_students: List[Student]) -> None:
    """Test identifying highest scoring student."""
    highest = get_highest_scorer(sample_students)
    assert highest.name == "Aisha"
    assert highest.marks == 92.0


def test_get_highest_scorer_empty_list_raises() -> None:
    """Test get_highest_scorer on empty list raises ValueError."""
    with pytest.raises(ValueError, match="empty student list"):
        get_highest_scorer([])


def test_get_lowest_scorer(sample_students: List[Student]) -> None:
    """Test identifying lowest scoring student."""
    lowest = get_lowest_scorer(sample_students)
    assert lowest.name == "Rohan"
    assert lowest.marks == 67.0


def test_get_lowest_scorer_empty_list_raises() -> None:
    """Test get_lowest_scorer on empty list raises ValueError."""
    with pytest.raises(ValueError, match="empty student list"):
        get_lowest_scorer([])


def test_count_students(sample_students: List[Student]) -> None:
    """Test counting total number of students."""
    assert count_students(sample_students) == 4
    assert count_students([]) == 0


def test_filter_students_by_marks(sample_students: List[Student]) -> None:
    """Test filtering students by minimum marks threshold."""
    filtered = filter_students_by_marks(sample_students, 80.0)
    assert len(filtered) == 2
    names = [s.name for s in filtered]
    assert "Rahul" in names
    assert "Aisha" in names


def test_get_performance_distribution(sample_students: List[Student]) -> None:
    """Test tallying performance distribution using Counter."""
    dist = get_performance_distribution(sample_students)
    assert dist[PerformanceLevel.EXCELLENT] == 1  # Aisha (92)
    assert dist[PerformanceLevel.GOOD] == 2       # Rahul (85), Sneha (78)
    assert dist[PerformanceLevel.AVERAGE] == 1    # Rohan (67)
    assert dist[PerformanceLevel.POOR] == 0


def test_performance_level_classification_boundaries() -> None:
    """Test get_performance_level classification boundaries [90+, 75-89, 50-74, <50]."""
    assert get_performance_level(95.0) == PerformanceLevel.EXCELLENT
    assert get_performance_level(90.0) == PerformanceLevel.EXCELLENT
    assert get_performance_level(85.0) == PerformanceLevel.GOOD
    assert get_performance_level(75.0) == PerformanceLevel.GOOD
    assert get_performance_level(65.0) == PerformanceLevel.AVERAGE
    assert get_performance_level(50.0) == PerformanceLevel.AVERAGE
    assert get_performance_level(45.0) == PerformanceLevel.POOR

    with pytest.raises(ValueError):
        get_performance_level(-10.0)
    with pytest.raises(ValueError):
        get_performance_level(150.0)
