# ==============================================================================
# Program    : Security & Cryptographic Utilities (security.py)
# Objective  : Implement password hashing, verification, and JWT access token creation/decoding.
# Concept    : One-Way Password Hashing & HMAC-SHA256 Signed JWT Tokens
# Why Used   : Encapsulates cryptographic operations in a clean, reusable module.
# ==============================================================================

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import jwt

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.config import settings

# Attempt importing pwdlib (Argon2), fallback to passlib (Bcrypt) if pwdlib unavailable
try:
    from pwdlib import PasswordHash
    _password_hash = PasswordHash.recommended()

    def hash_password(password: str) -> str:
        """Hash plaintext password using Argon2id."""
        return _password_hash.hash(password)

    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify plaintext candidate password against stored Argon2id hash."""
        try:
            return _password_hash.verify(plain_password, hashed_password)
        except Exception:
            return False

except Exception:
    from passlib.context import CryptContext
    _pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def hash_password(password: str) -> str:
        """Hash plaintext password using Bcrypt fallback."""
        return _pwd_context.hash(password)

    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify plaintext candidate password against stored Bcrypt hash."""
        try:
            return _pwd_context.verify(plain_password, hashed_password)
        except Exception:
            return False

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Generate cryptographically signed JWT access token containing claims and expiration timestamp."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": int(expire.timestamp())})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and validate JWT access token signature and expiration timestamp.

    Raises:
        jwt.PyJWTError: If token signature is invalid or expired.
    """
    payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    return payload
