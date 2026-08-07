# ==============================================================================
# Program    : Unit Test Using setUp() and tearDown() Fixtures
# Objective  : Verify file creation and contents using automated test setup and teardown fixtures.
# Concept    : Test Fixture Lifecycle (setUp & tearDown)
# Why Used   : setUp() creates test file before execution; tearDown() deletes it afterwards.
# ==============================================================================

import os
import unittest

class TestFileExistence(unittest.TestCase):

    # What is used : setUp() fixture method
    # Why it is used: Automatically executes before each test method to create test file
    def setUp(self):
        self.filename = "test_fixture_file.tmp"
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write("Hello Unit Testing!")

    # What is used : tearDown() fixture method
    # Why it is used: Automatically executes after each test method to delete temporary test file
    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_file_exists(self):
        self.assertTrue(os.path.exists(self.filename))

    def test_file_content(self):
        with open(self.filename, "r", encoding="utf-8") as f:
            data = f.read()
        self.assertEqual(data, "Hello Unit Testing!")

if __name__ == "__main__":
    unittest.main()
