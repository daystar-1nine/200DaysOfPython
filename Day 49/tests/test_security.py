# ==============================================================================
# Program    : Cryptographic Security Unit Tests & Parameterization (test_security.py)
# Objective  : Unit test hash_password, verify_password, create_access_token, decode_access_token, and input validation.
# Concept    : Unit Testing & Parameterized Tests (@pytest.mark.parametrize)
# Why Used   : Verifies security mechanics independently of API routing layers.
# ==============================================================================

import os
import sys
import pytest
from datetime import timedelta
import jwt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.security import hash_password, verify_password, create_access_token, decode_access_token
from app.config import settings

def test_password_is_hashed():
    """Verify plaintext password produces hashed output different from input string."""
    password = "StrongPassword123!"
    hashed = hash_password(password)
    assert hashed != password
    assert len(hashed) > 20

def test_correct_password_verifies():
    """Verify correct plaintext candidate verifies against stored hash."""
    password = "StrongPassword123!"
    hashed = hash_password(password)
    assert verify_password(password, hashed) is True

def test_wrong_password_fails():
    """Verify incorrect plaintext candidate fails verification."""
    password = "StrongPassword123!"
    hashed = hash_password(password)
    assert verify_password("WrongPassword123!", hashed) is False

def test_empty_password_verification_fails():
    """Verify empty string password candidate returns False."""
    hashed = hash_password("ValidPassword123!")
    assert verify_password("", hashed) is False

def test_jwt_create_and_decode_token():
    """Verify JWT access token creation and decoding of sub, email, and role claims."""
    claims = {"sub": "42", "email": "test@example.com", "role": "user"}
    token = create_access_token(data=claims)
    decoded = decode_access_token(token)

    assert decoded["sub"] == "42"
    assert decoded["email"] == "test@example.com"
    assert decoded["role"] == "user"
    assert "exp" in decoded

def test_jwt_expired_token_raises_exception():
    """Verify decoding an expired JWT token raises ExpiredSignatureError."""
    claims = {"sub": "42"}
    expired_token = create_access_token(data=claims, expires_delta=timedelta(seconds=-10))

    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(expired_token)

def test_jwt_invalid_signature_raises_exception():
    """Verify token signed with wrong secret key raises InvalidSignatureError."""
    claims = {"sub": "42"}
    token = create_access_token(data=claims)
    wrong_key = "different_secret_key_12345"

    with pytest.raises(jwt.InvalidSignatureError):
        jwt.decode(token, wrong_key, algorithms=[settings.JWT_ALGORITHM])

@pytest.mark.parametrize("invalid_email", [
    "invalidemail",
    "plainaddress",
    "@missingusername.com",
    "username@.com",
    "username@domain..com"
])
def test_parameterized_invalid_emails(client, invalid_email):
    """Parameterized test verifying registration rejects malformed email strings (422)."""
    payload = {
        "name": "Invalid Email Test",
        "email": invalid_email,
        "password": "Password123!"
    }
    response = client.post("/auth/register", json=payload)
    assert response.status_code == 422

@pytest.mark.parametrize("invalid_password", [
    "123",
    "a",
    "short"
])
def test_parameterized_too_short_passwords(client, invalid_password):
    """Parameterized test verifying registration rejects passwords shorter than 6 characters (422)."""
    payload = {
        "name": "Short Password Test",
        "email": f"valid_{invalid_password}@example.com",
        "password": invalid_password
    }
    response = client.post("/auth/register", json=payload)
    assert response.status_code == 422
