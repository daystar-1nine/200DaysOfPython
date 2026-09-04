# ==============================================================================
# Program    : Authentication Integration Tests (test_auth.py)
# Objective  : Test POST /auth/register, POST /auth/login, credential verification, and error responses.
# Concept    : API Integration Testing
# Why Used   : Validates registration, authentication, and HTTP error response contracts.
# ==============================================================================

import os
import sys
from datetime import timedelta
from app.security import create_access_token

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_register_user_success(client):
    """Test successful user registration returns 201 Created and user metadata excluding password_hash."""
    payload = {
        "name": "New Registration",
        "email": "new.user@example.com",
        "password": "SecurePassword123!",
        "age": 22
    }
    response = client.post("/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "New Registration"
    assert data["email"] == "new.user@example.com"
    assert "id" in data
    assert "password_hash" not in data

def test_register_duplicate_email_returns_409(client, normal_user):
    """Test registering with an existing email returns HTTP 409 Conflict with standardized error structure."""
    payload = {
        "name": "Duplicate User",
        "email": normal_user.email,
        "password": "Password123!"
    }
    response = client.post("/auth/register", json=payload)
    assert response.status_code == 409
    data = response.json()
    assert data["error"]["code"] == "DUPLICATE_EMAIL"

def test_login_success(client, normal_user):
    """Test logging in with valid credentials returns 200 OK and signed bearer token."""
    payload = {
        "email": normal_user.email,
        "password": "UserPassword123!"
    }
    response = client.post("/auth/login", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_login_wrong_password_returns_401(client, normal_user):
    """Test logging in with wrong password returns HTTP 401 Unauthorized."""
    payload = {
        "email": normal_user.email,
        "password": "WrongPassword123!"
    }
    response = client.post("/auth/login", json=payload)
    assert response.status_code == 401
    data = response.json()
    assert data["error"]["code"] == "INVALID_CREDENTIALS"

def test_login_unknown_email_returns_401(client):
    """Test logging in with unregistered email returns HTTP 401 Unauthorized."""
    payload = {
        "email": "nonexistent@example.com",
        "password": "Password123!"
    }
    response = client.post("/auth/login", json=payload)
    assert response.status_code == 401
    data = response.json()
    assert data["error"]["code"] == "INVALID_CREDENTIALS"

def test_access_protected_endpoint_without_token_returns_401(client):
    """Test requesting protected endpoint without Authorization header returns HTTP 401 Unauthorized."""
    response = client.get("/users/me")
    assert response.status_code == 401

def test_access_protected_endpoint_with_invalid_token_returns_401(client):
    """Test requesting protected endpoint with garbage token string returns HTTP 401 Unauthorized."""
    headers = {"Authorization": "Bearer invalid_token_garbage_string"}
    response = client.get("/users/me", headers=headers)
    assert response.status_code == 401

def test_access_protected_endpoint_with_expired_token_returns_401(client, normal_user):
    """Test requesting protected endpoint with expired token returns HTTP 401 Unauthorized."""
    expired_token = create_access_token(
        data={"sub": str(normal_user.id), "email": normal_user.email},
        expires_delta=timedelta(seconds=-60)
    )
    headers = {"Authorization": f"Bearer {expired_token}"}
    response = client.get("/users/me", headers=headers)
    assert response.status_code == 401
    data = response.json()
    assert data["error"]["code"] == "AUTHENTICATION_FAILED"
