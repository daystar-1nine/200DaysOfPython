# ==============================================================================
# Test Suite : Authentication & Security Tests (test_auth.py)
# Objective  : Test registration, password hashing, login, JWT validation, expiration, and invalid credentials.
# Concept    : Security Integration Testing
# Why Used   : Verifies security mechanics and credential verification.
# ==============================================================================

import os
import sys
from datetime import timedelta
import pytest

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.security import hash_password, verify_password, create_access_token, decode_access_token

def test_password_hashing_and_verification():
    password = "MySecurePassword123!"
    hashed = hash_password(password)
    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword123!", hashed) is False

def test_register_user_success(client):
    response = client.post("/auth/register", json={
        "name": "Suraj",
        "email": "suraj_reg@example.com",
        "password": "Password123!",
        "age": 21
    })
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Suraj"
    assert data["email"] == "suraj_reg@example.com"
    assert "password_hash" not in data  # Sensitive field NEVER exposed

def test_register_duplicate_email_fails(client):
    client.post("/auth/register", json={
        "name": "User One",
        "email": "dup@example.com",
        "password": "Password123!"
    })
    response = client.post("/auth/register", json={
        "name": "User Two",
        "email": "dup@example.com",
        "password": "Password123!"
    })
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]

def test_login_success(client):
    client.post("/auth/register", json={
        "name": "Suraj Login",
        "email": "login@example.com",
        "password": "SecretPassword123!"
    })
    response = client.post("/auth/login", json={
        "email": "login@example.com",
        "password": "SecretPassword123!"
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_login_invalid_password_fails(client):
    client.post("/auth/register", json={
        "name": "Suraj Login",
        "email": "wrongpass@example.com",
        "password": "SecretPassword123!"
    })
    response = client.post("/auth/login", json={
        "email": "wrongpass@example.com",
        "password": "IncorrectPassword!"
    })
    assert response.status_code == 401
    assert "Invalid email or password" in response.json()["detail"]

def test_login_nonexistent_email_fails(client):
    response = client.post("/auth/login", json={
        "email": "nonexistent@example.com",
        "password": "SomePassword123!"
    })
    assert response.status_code == 401

def test_jwt_access_token_creation_and_decoding():
    payload = {"sub": "42", "email": "test@example.com", "role": "user"}
    token = create_access_token(payload)
    decoded = decode_access_token(token)
    assert decoded["sub"] == "42"
    assert decoded["email"] == "test@example.com"
    assert "exp" in decoded

def test_expired_jwt_token_fails(client):
    token = create_access_token({"sub": "1"}, expires_delta=timedelta(seconds=-10))
    response = client.get("/users/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()

def test_invalid_jwt_signature_fails(client):
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalidpayload.invalidsignature"
    response = client.get("/users/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401

def test_missing_auth_header_fails(client):
    response = client.get("/users/me")
    assert response.status_code == 401
