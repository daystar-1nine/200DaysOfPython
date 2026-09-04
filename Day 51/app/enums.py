"""
===============================================================================
DAY 51 — PERFORMANCE LEVEL ENUM DEFINITION & CLASSIFICATION LOGIC
===============================================================================
This module defines the PerformanceLevel Enum and classification utility.
===============================================================================
"""

from enum import Enum


class PerformanceLevel(str, Enum):
    """Enumeration of academic performance classification tiers."""

    EXCELLENT = "Excellent"
    GOOD = "Good"
    AVERAGE = "Average"
    POOR = "Poor"


def get_performance_level(marks: float) -> PerformanceLevel:
    """Classify student academic marks into a PerformanceLevel Enum member."""
    # What is used: Multi-branch conditional range evaluation.
    # Why it is used: Categorizes numeric marks into predefined enum state buckets.
    # How it works: Returns EXCELLENT for 90+, GOOD for 75-89, AVERAGE for 50-74, POOR for <50.
    if marks < 0 or marks > 100:
        raise ValueError(f"Invalid marks value: {marks}. Marks must be between 0 and 100.")

    if marks >= 90.0:
        return PerformanceLevel.EXCELLENT
    elif marks >= 75.0:
        return PerformanceLevel.GOOD
    elif marks >= 50.0:
        return PerformanceLevel.AVERAGE
    else:
        return PerformanceLevel.POOR
