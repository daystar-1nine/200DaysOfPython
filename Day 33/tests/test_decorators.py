# ==============================================================================
# Test Suite : Day 33 Function Monitoring System Pytest Suite
# Objective  : Test 12+ scenarios for @logger, @timer, @retry, and @requires_auth.
# Concept    : Unit Testing Python Decorators & Metadata Preservation
# Why Used   : Asserts decorator behavior, metadata preservation, and exception handling.
# ==============================================================================

import os
import sys
import unittest
import pytest

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from decorators.logger import logger
from decorators.timer import timer
from decorators.retry import retry
from decorators.auth import requires_auth, CURRENT_USER

# 1. Test Metadata Preservation (__name__, __doc__)
def test_metadata_preservation():
    @logger
    def sample_func():
        """Sample docstring."""
        return 42
    assert sample_func.__name__ == "sample_func"
    assert sample_func.__doc__ == "Sample docstring."

# 2. Test Function Return Value
def test_return_value():
    @logger
    def add(a, b):
        return a + b
    assert add(10, 20) == 30

# 3. Test Positional Arguments
def test_positional_arguments():
    @logger
    def multiply(a, b, c):
        return a * b * c
    assert multiply(2, 3, 4) == 24

# 4. Test Keyword Arguments
def test_keyword_arguments():
    @logger
    def greet(name, msg="Hello"):
        return f"{msg}, {name}!"
    assert greet(name="Suraj", msg="Welcome") == "Welcome, Suraj!"

# 5. Test Exception Propagation
def test_exception_propagation():
    @logger
    def failing_func():
        raise ValueError("Something went wrong")
    with pytest.raises(ValueError, match="Something went wrong"):
        failing_func()

# 6. Test Timer Execution Time
def test_timer_decorator():
    @timer
    def slow_func():
        return "done"
    res = slow_func()
    assert res == "done"

# 7. Test Retry Success on Final Attempt
def test_retry_success():
    attempts = 0
    @retry(max_attempts=3, delay=0.0)
    def flaky_func():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("Flaky error")
        return "recovered"
    
    assert flaky_func() == "recovered"
    assert attempts == 3

# 8. Test Retry Exhaustion Raises Last Exception
def test_retry_exhaustion():
    attempts = 0
    @retry(max_attempts=2, delay=0.0)
    def always_fails():
        nonlocal attempts
        attempts += 1
        raise KeyError("Persistent failure")
    
    with pytest.raises(KeyError, match="Persistent failure"):
        always_fails()
    assert attempts == 2

# 9. Test Authorization Success
def test_auth_success():
    CURRENT_USER["is_authenticated"] = True
    CURRENT_USER["role"] = "admin"
    @requires_auth(role="admin")
    def admin_area():
        return "admin_ok"
    assert admin_area() == "admin_ok"

# 10. Test Authorization Failure (Not Authenticated)
def test_auth_unauthenticated():
    CURRENT_USER["is_authenticated"] = False
    @requires_auth(role="admin")
    def protected():
        return "ok"
    with pytest.raises(PermissionError, match="not authenticated"):
        protected()
    CURRENT_USER["is_authenticated"] = True

# 11. Test Authorization Failure (Wrong Role)
def test_auth_wrong_role():
    CURRENT_USER["is_authenticated"] = True
    CURRENT_USER["role"] = "user"
    @requires_auth(role="admin")
    def admin_only():
        return "ok"
    with pytest.raises(PermissionError, match="Requires 'admin' role"):
        admin_only()
    CURRENT_USER["role"] = "admin"

# 12. Test Stacked Decorators
def test_stacked_decorators():
    @timer
    @logger
    def stacked_func(x):
        return x * 2
    assert stacked_func(5) == 10

class TestDecoratorsRunner(unittest.TestCase):
    def test_decorators_standalone(self):
        @logger
        def test_sub(a, b):
            return a - b
        self.assertEqual(test_sub(10, 4), 6)

if __name__ == "__main__":
    unittest.main()
