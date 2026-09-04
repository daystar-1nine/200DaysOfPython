"""
===============================================================================
DAY 50 — SECURITY UTILITIES (PASSWORD HASHING & JWT SIGNING)
===============================================================================
This module provides password hashing and verification using pwdlib / passlib,
alongside PyJWT token generation and verification utilities.
===============================================================================
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict
import jwt
from app.config import settings

# Attempt using pwdlib if available, fallback to passlib pbkdf2_sha256
try:
    from pwdlib import PasswordHash
    _password_hash = PasswordHash.recommended()

    def hash_password(password: str) -> str:
        """Hash cleartext password using pwdlib recommended algorithm."""
        return _password_hash.hash(password)

    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify plain password against stored digest."""
        try:
            return _password_hash.verify(plain_password, hashed_password)
        except Exception:
            return False

except Exception:
    from passlib.context import CryptContext
    _pwd_context = CryptContext(schemes=["pbkdf2_sha256", "bcrypt"], deprecated="auto")

    def hash_password(password: str) -> str:
        """Hash cleartext password using passlib pbkdf2_sha256."""
        # Truncate password to 72 chars if using bcrypt
        return _pwd_context.hash(password[:72])

    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify plain password against stored hash."""
        try:
            return _pwd_context.verify(plain_password[:72], hashed_password)
        except Exception:
            return False


def create_access_token(data: Dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT access token containing claims payload."""
    # What is used: Pyjwt encode utility with timestamp expiration payload.
    # Why it is used: Issues signed bearer token to client after successful login.
    # How it works: Appends exp claim and signs token with settings.SECRET_KEY.
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": int(expire.timestamp())})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and verify signature/expiration of a JWT access token."""
    # What is used: Pyjwt decode utility.
    # Why it is used: Verifies token signature and extracts claims payload.
    # How it works: Decodes token using secret key; raises ExpiredSignatureError or PyJWTError if invalid.
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
