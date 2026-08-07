# ==============================================================================
# Program    : Unit Test for String Reversal Function
# Objective  : Verify string reversal logic for single words, sentences, and empty strings.
# Concept    : String Manipulation Testing
# Why Used   : Ensures string reversal preserves character ordering accurately.
# ==============================================================================

import unittest

def reverse_string(s):
    """Returns the reversed version of string s."""
    return s[::-1]

class TestStringReverse(unittest.TestCase):

    def test_reverse_normal_word(self):
        self.assertEqual(reverse_string("python"), "nohtyp")

    def test_reverse_empty_string(self):
        self.assertEqual(reverse_string(""), "")

    def test_reverse_palindrome(self):
        self.assertEqual(reverse_string("racecar"), "racecar")

if __name__ == "__main__":
    unittest.main()
