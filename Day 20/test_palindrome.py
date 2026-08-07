# ==============================================================================
# Program    : Unit Test for Palindrome Checker Function
# Objective  : Verify case-insensitive palindrome detection logic.
# Concept    : String Normalization & Boolean Predicates
# Why Used   : Asserts that strings like 'madam' or 'RaceCar' return True, while 'python' returns False.
# ==============================================================================

import unittest

def is_palindrome(text):
    """Returns True if normalized text is a palindrome."""
    clean_text = "".join(ch.lower() for ch in text if ch.isalnum())
    return clean_text == clean_text[::-1]

class TestPalindrome(unittest.TestCase):

    def test_valid_palindromes(self):
        self.assertTrue(is_palindrome("madam"))
        self.assertTrue(is_palindrome("RaceCar"))
        self.assertTrue(is_palindrome("A man, a plan, a canal: Panama"))

    def test_invalid_palindromes(self):
        self.assertFalse(is_palindrome("python"))
        self.assertFalse(is_palindrome("hello world"))

if __name__ == "__main__":
    unittest.main()
