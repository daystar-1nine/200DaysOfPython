# ==============================================================================
# Program    : Unit Test for List Sorting Function
# Objective  : Verify ascending and descending list sorting operations.
# Concept    : Container Assertions & Sequence Ordering
# Why Used   : Asserts that sorted list matches expected ordered sequences.
# ==============================================================================

import unittest

def sort_numbers(numbers, reverse=False):
    """Returns a new sorted list of numbers."""
    return sorted(numbers, reverse=reverse)

class TestListSorting(unittest.TestCase):

    def test_sort_ascending(self):
        nums = [5, 2, 8, 1, 9]
        self.assertEqual(sort_numbers(nums), [1, 2, 5, 8, 9])

    def test_sort_descending(self):
        nums = [5, 2, 8, 1, 9]
        self.assertEqual(sort_numbers(nums, reverse=True), [9, 8, 5, 2, 1])

    def test_sort_empty_and_single(self):
        self.assertEqual(sort_numbers([]), [])
        self.assertEqual(sort_numbers([42]), [42])

if __name__ == "__main__":
    unittest.main()
