"""
===============================================================================
DAY 50 — AUTHENTICATION INTEGRATION TESTS
===============================================================================
This module tests registration, login, duplicate email checks, case insensitivity,
and token validation.
===============================================================================
"""

from datetime import timedelta
from fastapi.testclient import TestClient
from app.security import create_access_token


def test_register_user_success(client: TestClient) -> None:
    """Test successful user registration (POST /auth/register)."""
    payload = {
        "name": "Suraj Sawant",
        "email": "newuser@example.com",
        "password": "securepassword123",
        "role": "user",
    }
    response = client.post("/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "newuser@example.com"
    assert data["name"] == "Suraj Sawant"
    assert "id" in data
    assert "password" not in data
    assert "password_hash" not in data


def test_register_duplicate_email_fails(client: TestClient) -> None:
    """Test registration with existing email returns 409 Conflict."""
    payload = {
        "name": "First User",
        "email": "duplicate@example.com",
        "password": "password123",
    }
    res1 = client.post("/auth/register", json=payload)
    assert res1.status_code == 201

    res2 = client.post("/auth/register", json=payload)
    assert res2.status_code == 409
    assert res2.json()["error"]["code"] == "CONFLICT"


def test_register_invalid_email_format_fails(client: TestClient) -> None:
    """Test registration with invalid email format returns 422 Unprocessable Entity."""
    payload = {
        "name": "Invalid Email",
        "email": "not-an-email",
        "password": "password123",
    }
    response = client.post("/auth/register", json=payload)
    assert response.status_code == 422


def test_register_short_password_fails(client: TestClient) -> None:
    """Test registration with short password returns 422 validation error."""
    payload = {
        "name": "Short Pass",
        "email": "short@example.com",
        "password": "123",
    }
    response = client.post("/auth/register", json=payload)
    assert response.status_code == 422


def test_login_success(client: TestClient) -> None:
    """Test successful user login returning JWT access token."""
    reg_payload = {
        "name": "Login User",
        "email": "login@example.com",
        "password": "mypassword123",
    }
    client.post("/auth/register", json=reg_payload)

    login_payload = {
        "email": "login@example.com",
        "password": "mypassword123",
    }
    response = client.post("/auth/login", json=login_payload)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_case_insensitive_email(client: TestClient) -> None:
    """Test login works with uppercase/mixed-case email."""
    reg_payload = {
        "name": "Case User",
        "email": "caseuser@example.com",
        "password": "mypassword123",
    }
    client.post("/auth/register", json=reg_payload)

    login_payload = {
        "email": "CASEUSER@EXAMPLE.COM",
        "password": "mypassword123",
    }
    response = client.post("/auth/login", json=login_payload)
    assert response.status_code == 200


def test_login_wrong_password_fails(client: TestClient) -> None:
    """Test login with incorrect password returns 401 Unauthorized."""
    reg_payload = {
        "name": "User One",
        "email": "userone@example.com",
        "password": "correctpassword",
    }
    client.post("/auth/register", json=reg_payload)

    login_payload = {
        "email": "userone@example.com",
        "password": "wrongpassword",
    }
    response = client.post("/auth/login", json=login_payload)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


def test_login_nonexistent_email_fails(client: TestClient) -> None:
    """Test login with non-existent email returns 401 Unauthorized."""
    login_payload = {
        "email": "nonexistent@example.com",
        "password": "somepassword",
    }
    response = client.post("/auth/login", json=login_payload)
    assert response.status_code == 401


def test_missing_authorization_header_fails(client: TestClient) -> None:
    """Test calling protected endpoint without bearer header returns 401."""
    response = client.get("/users/me")
    assert response.status_code == 401


def test_invalid_token_format_fails(client: TestClient) -> None:
    """Test calling protected endpoint with invalid bearer token returns 401."""
    headers = {"Authorization": "Bearer invalid_jwt_token_string"}
    response = client.get("/users/me", headers=headers)
    assert response.status_code == 401


def test_expired_token_header_fails(client: TestClient) -> None:
    """Test calling protected endpoint with expired token returns 401."""
    expired_token = create_access_token({"sub": "1"}, expires_delta=timedelta(seconds=-10))
    headers = {"Authorization": f"Bearer {expired_token}"}
    response = client.get("/users/me", headers=headers)
    assert response.status_code == 401
