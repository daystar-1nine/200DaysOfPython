# ==============================================================================
# Program    : Domain Exception & Standardized Error Response Tests (test_errors.py)
# Objective  : Test custom domain exceptions, global exception handlers, and standardized JSON error payloads.
# Concept    : Global Exception Handling & Error Consistency Testing
# Why Used   : Guarantees every API error returns predictable code, message, and request_id properties.
# ==============================================================================

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.exceptions import (
    UserNotFoundError,
    DuplicateEmailError,
    InvalidCredentialsError,
    AuthenticationError,
    AuthorizationError,
    ProductNotFoundError,
    OrderNotFoundError,
    InsufficientStockError,
    PaymentGatewayError
)

def test_user_not_found_exception_properties():
    """Unit test verifying UserNotFoundError properties (404 status code & code="USER_NOT_FOUND")."""
    exc = UserNotFoundError(99)
    assert exc.status_code == 404
    assert exc.code == "USER_NOT_FOUND"
    assert "99" in exc.message

def test_duplicate_email_exception_properties():
    """Unit test verifying DuplicateEmailError properties (409 status code & code="DUPLICATE_EMAIL")."""
    exc = DuplicateEmailError("test@example.com")
    assert exc.status_code == 409
    assert exc.code == "DUPLICATE_EMAIL"

def test_insufficient_stock_exception_properties():
    """Unit test verifying InsufficientStockError properties (409 status code & code="INSUFFICIENT_STOCK")."""
    exc = InsufficientStockError("Laptop", 5, 1)
    assert exc.status_code == 409
    assert exc.code == "INSUFFICIENT_STOCK"

def test_standardized_error_format_user_not_found(client):
    """Integration test verifying GET /products/99999 returns standardized JSON error response structure."""
    response = client.get("/products/99999")
    assert response.status_code == 404
    data = response.json()

    assert "error" in data
    err = data["error"]
    assert err["code"] == "PRODUCT_NOT_FOUND"
    assert "99999" in err["message"]
    assert "request_id" in err

def test_standardized_error_format_duplicate_email(client, normal_user):
    """Integration test verifying duplicate user registration returns standardized JSON error (409 DUPLICATE_EMAIL)."""
    payload = {
        "name": "Duplicate User",
        "email": normal_user.email,
        "password": "Password123!"
    }
    response = client.post("/auth/register", json=payload)
    assert response.status_code == 409
    data = response.json()

    assert "error" in data
    err = data["error"]
    assert err["code"] == "DUPLICATE_EMAIL"
    assert "already exists" in err["message"]

def test_standardized_error_format_invalid_credentials(client):
    """Integration test verifying login failure returns standardized JSON error (401 INVALID_CREDENTIALS)."""
    payload = {"email": "unknown@example.com", "password": "WrongPassword123!"}
    response = client.post("/auth/login", json=payload)
    assert response.status_code == 401
    data = response.json()

    assert "error" in data
    err = data["error"]
    assert err["code"] == "INVALID_CREDENTIALS"

def test_standardized_error_format_forbidden_role(client, normal_user_headers):
    """Integration test verifying non-admin calling /admin/users returns standardized JSON error (403 FORBIDDEN)."""
    response = client.get("/admin/users", headers=normal_user_headers)
    assert response.status_code == 403
    data = response.json()

    assert "error" in data
    err = data["error"]
    assert err["code"] == "FORBIDDEN"

def test_standardized_error_format_validation_error(client):
    """Integration test verifying invalid payload returns standardized JSON error (422 VALIDATION_ERROR) with field details."""
    payload = {"name": "Test", "email": "invalid_email_format", "password": "123"}
    response = client.post("/auth/register", json=payload)
    assert response.status_code == 422
    data = response.json()

    assert "error" in data
    err = data["error"]
    assert err["code"] == "VALIDATION_ERROR"
    assert "fields" in err
    assert err["fields"] is not None
