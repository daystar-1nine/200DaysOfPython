# ==============================================================================
# Test Suite : Day 37 Custom Exceptions Pytest Suite (16 Test Cases)
# Objective  : Test every custom exception class, codes, inheritance, and exception chaining.
# Concept    : Unit Testing Custom Exception Hierarchies
# Why Used   : Asserts error taxonomy, code attributes, and cause preservation.
# ==============================================================================

import os
import sys
import unittest
import pytest

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from exceptions import (
    ApplicationError,
    ValidationError,
    BoundsError,
    FormatError,
    DatabaseError,
    UniqueConstraintError,
    NotFoundError,
    ExternalServiceError,
    TimeoutError,
    AuthenticationError
)
from app_service import UserService

# 1. Test ApplicationError base attributes
def test_application_error_base():
    err = ApplicationError("Base error", code="TEST_CODE")
    assert str(err) == "[TEST_CODE] Base error"
    assert err.message == "Base error"
    assert err.code == "TEST_CODE"

# 2. Test ValidationError hierarchy
def test_validation_error():
    err = ValidationError("Invalid input")
    assert isinstance(err, ApplicationError)
    assert err.code == "VALIDATION_ERROR"

# 3. Test BoundsError hierarchy
def test_bounds_error():
    err = BoundsError("Out of bounds")
    assert isinstance(err, ValidationError)
    assert err.code == "BOUNDS_ERROR"

# 4. Test FormatError hierarchy
def test_format_error():
    err = FormatError("Invalid format")
    assert isinstance(err, ValidationError)
    assert err.code == "FORMAT_ERROR"

# 5. Test DatabaseError hierarchy
def test_database_error():
    err = DatabaseError("Query failed")
    assert isinstance(err, ApplicationError)
    assert err.code == "DATABASE_ERROR"

# 6. Test UniqueConstraintError hierarchy
def test_unique_constraint_error():
    err = UniqueConstraintError("Duplicate key")
    assert isinstance(err, DatabaseError)
    assert err.code == "UNIQUE_CONSTRAINT_ERROR"

# 7. Test NotFoundError hierarchy
def test_not_found_error():
    err = NotFoundError("Record missing")
    assert isinstance(err, ApplicationError)
    assert err.code == "NOT_FOUND_ERROR"

# 8. Test ExternalServiceError hierarchy
def test_external_service_error():
    err = ExternalServiceError("API down")
    assert isinstance(err, ApplicationError)
    assert err.code == "EXTERNAL_SERVICE_ERROR"

# 9. Test TimeoutError hierarchy
def test_timeout_error():
    err = TimeoutError("Request timed out")
    assert isinstance(err, ExternalServiceError)
    assert err.code == "TIMEOUT_ERROR"

# 10. Test AuthenticationError hierarchy
def test_authentication_error():
    err = AuthenticationError("Unauthorized")
    assert isinstance(err, ApplicationError)
    assert err.code == "AUTHENTICATION_ERROR"

# 11. Test Exception Chaining Cause Preservation
def test_exception_chaining_cause():
    try:
        try:
            int("invalid")
        except ValueError as cause:
            raise ValidationError("Parsing integer failed") from cause
    except ValidationError as err:
        assert isinstance(err.__cause__, ValueError)

# 12. Test Catching All Exceptions via ApplicationError
def test_catch_all_application_errors():
    errors = [
        ValidationError("v"),
        DatabaseError("d"),
        NotFoundError("n"),
        ExternalServiceError("e")
    ]
    for e in errors:
        assert isinstance(e, ApplicationError)

# 13. Test UserService Validation Error
def test_user_service_invalid_email(tmp_path):
    db_file = str(tmp_path / "test_val.db")
    service = UserService(db_file)
    with pytest.raises(ValidationError, match="Invalid email format"):
        service.register_user("no_at_sign", 25)

# 14. Test UserService Bounds Error
def test_user_service_bounds_error(tmp_path):
    db_file = str(tmp_path / "test_bounds.db")
    service = UserService(db_file)
    with pytest.raises(BoundsError, match="out of permitted registration bounds"):
        service.register_user("suraj@example.com", 15)

# 15. Test UserService UniqueConstraintError with Chaining
def test_user_service_duplicate_email(tmp_path):
    db_file = str(tmp_path / "test_dup.db")
    service = UserService(db_file)
    service.register_user("suraj@example.com", 25)
    with pytest.raises(UniqueConstraintError) as cm:
        service.register_user("suraj@example.com", 30)
    assert cm.value.__cause__ is not None  # Chained sqlite3.IntegrityError

# 16. Test UserService NotFoundError
def test_user_service_not_found(tmp_path):
    db_file = str(tmp_path / "test_nf.db")
    service = UserService(db_file)
    with pytest.raises(NotFoundError, match="was not found"):
        service.get_user_by_id(9999)

class TestExceptionsRunner(unittest.TestCase):
    def test_exceptions_standalone(self):
        err = ValidationError("Standalone test")
        self.assertEqual(err.code, "VALIDATION_ERROR")

if __name__ == "__main__":
    unittest.main()
