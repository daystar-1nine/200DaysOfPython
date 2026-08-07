# ==============================================================================
# Program    : Student Grade System & Test Suite (Challenge Project)
# Objective  : Implement grade calculation functions and test with boundary and invalid values.
# Concept    : Boundary Value Analysis & Exception Testing
# Why Used   : Validates grade assignments (A, B, C, D, F), averages, and pass/fail thresholds.
# ==============================================================================

import unittest

# --- STUDENT GRADE SYSTEM MODULE ---
def calculate_average(marks_list):
    """Calculates average of marks list. Raises ValueError if empty or invalid."""
    if not marks_list:
        raise ValueError("Marks list cannot be empty.")
    for mark in marks_list:
        if not (0 <= mark <= 100):
            raise ValueError(f"Invalid mark value: {mark}. Marks must be between 0 and 100.")
    return sum(marks_list) / len(marks_list)

def calculate_grade(average):
    """Returns letter grade based on numerical average."""
    if average >= 90:
        return "A"
    elif average >= 80:
        return "B"
    elif average >= 70:
        return "C"
    elif average >= 60:
        return "D"
    else:
        return "F"

def is_pass(marks_list, passing_mark=40):
    """Returns True if all marks meet passing threshold, False otherwise."""
    if not marks_list:
        raise ValueError("Marks list cannot be empty.")
    return all(mark >= passing_mark for mark in marks_list)


# --- UNIT TEST SUITE ---
class TestStudentGradeSystem(unittest.TestCase):

    def test_calculate_average_valid(self):
        self.assertEqual(calculate_average([80, 90, 100]), 90.0)
        self.assertEqual(calculate_average([70, 75, 80]), 75.0)

    def test_calculate_average_empty_list_exception(self):
        with self.assertRaises(ValueError):
            calculate_average([])

    def test_calculate_average_out_of_bounds_exception(self):
        with self.assertRaises(ValueError):
            calculate_average([85, 105, 90])
        with self.assertRaises(ValueError):
            calculate_average([-10, 80, 90])

    def test_calculate_grade_boundaries(self):
        self.assertEqual(calculate_grade(95), "A")
        self.assertEqual(calculate_grade(90), "A")
        self.assertEqual(calculate_grade(85), "B")
        self.assertEqual(calculate_grade(80), "B")
        self.assertEqual(calculate_grade(75), "C")
        self.assertEqual(calculate_grade(70), "C")
        self.assertEqual(calculate_grade(65), "D")
        self.assertEqual(calculate_grade(60), "D")
        self.assertEqual(calculate_grade(55), "F")

    def test_is_pass_status(self):
        self.assertTrue(is_pass([50, 60, 70, 45]))
        self.assertFalse(is_pass([50, 60, 35, 80]))

if __name__ == "__main__":
    unittest.main()
