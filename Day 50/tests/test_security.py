"""
===============================================================================
DAY 50 — SECURITY & CRYPTOGRAPHY UNIT TESTS
===============================================================================
This module tests bcrypt password hashing/verification and JWT encoding/decoding.
===============================================================================
"""

from datetime import timedelta
import jwt
import pytest
from app.security import hash_password, verify_password, create_access_token, decode_access_token


def test_hash_password_and_verify() -> None:
    """Test hashing plain password and verifying against digest digest."""
    plain = "mySecretPassword123"
    hashed = hash_password(plain)
    assert hashed != plain
    assert verify_password(plain, hashed) is True
    assert verify_password("wrongPassword", hashed) is False


def test_create_and_decode_access_token() -> None:
    """Test encoding JWT access token and decoding claim payload."""
    payload = {"sub": "100", "role": "admin"}
    token = create_access_token(payload, expires_delta=timedelta(minutes=15))
    decoded = decode_access_token(token)
    assert decoded["sub"] == "100"
    assert decoded["role"] == "admin"
    assert "exp" in decoded


def test_expired_access_token_raises() -> None:
    """Test expired JWT access token throws ExpiredSignatureError."""
    payload = {"sub": "100"}
    token = create_access_token(payload, expires_delta=timedelta(seconds=-10))
    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(token)
