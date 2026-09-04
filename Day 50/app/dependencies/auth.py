"""
===============================================================================
DAY 50 — SECURITY & AUTHENTICATION DEPENDENCIES
===============================================================================
This module provides FastAPI Depends security dependencies for OAuth2 Bearer
token parsing, current user resolution, and Role-Based Access Control (RBAC).
===============================================================================
"""

import jwt
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.security import decode_access_token
from app.exceptions import AuthenticationError, ForbiddenError

# What is used: OAuth2PasswordBearer security scheme.
# Why it is used: Extracts Bearer token string automatically from Authorization HTTP request header.
# How it works: Reads Authorization header matching 'Bearer <token>'.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """FastAPI Dependency resolving current authenticated User from JWT token."""
    # What is used: JWT decoding and UserRepository user lookup.
    # Why it is used: Authenticates incoming request and injects current User model instance into route handler.
    # How it works: Decodes sub claim from token, fetches User from DB; raises AuthenticationError (401) on error.
    try:
        payload = decode_access_token(token)
        user_id_raw: str = payload.get("sub")
        if user_id_raw is None:
            raise AuthenticationError("Invalid token payload: missing sub claim.")
        user_id = int(user_id_raw)
    except jwt.PyJWTError:
        raise AuthenticationError("Invalid or expired authentication token.")

    user_repo = UserRepository(db)
    user = user_repo.get_by_id(user_id)
    if not user:
        raise AuthenticationError("Authenticated user account no longer exists.")

    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """FastAPI Dependency enforcing Admin role authorization (RBAC)."""
    # What is used: Role check evaluation on current_user instance.
    # Why it is used: Protects administrative endpoints from non-admin users.
    # How it works: Verifies current_user.role == "admin"; raises ForbiddenError (403) if false.
    if current_user.role != "admin":
        raise ForbiddenError("Administrator privileges required for this endpoint.")
    return current_user
