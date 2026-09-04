"""
===============================================================================
DAY 50 — AUTHENTICATION SERVICE (BUSINESS LOGIC LAYER)
===============================================================================
This module handles user registration, password verification, token issuance,
and authentication validation routines.
===============================================================================
"""

from app.models.user import User
from app.schemas.user import UserCreate, UserLogin, Token
from app.repositories.user_repository import UserRepository
from app.security import hash_password, verify_password, create_access_token
from app.exceptions import ConflictError, AuthenticationError
from app.logging_config import logger


class AuthService:
    """Service orchestrating authentication and user account creation."""

    def __init__(self, user_repo: UserRepository) -> None:
        self.user_repo = user_repo

    def register(self, payload: UserCreate) -> User:
        """Register a new user account."""
        # What is used: Email duplicate check and password hashing.
        # Why it is used: Ensures email uniqueness and hashes raw cleartext password.
        # How it works: Checks repo for existing email; if found, raises ConflictError (409).
        existing = self.user_repo.get_by_email(payload.email)
        if existing:
            logger.warning(f"Registration failed. Email '{payload.email}' already registered.")
            raise ConflictError(f"User with email '{payload.email}' already exists.")

        user = User(
            name=payload.name,
            email=payload.email.lower(),
            password_hash=hash_password(payload.password),
            role=payload.role,
        )
        created_user = self.user_repo.create(user)
        logger.info({"event": "user_registered", "user_id": created_user.id, "email": created_user.email})
        return created_user

    def login(self, payload: UserLogin) -> Token:
        """Authenticate user credentials and return JWT access token."""
        # What is used: Password verification and token creation.
        # Why it is used: Validates user identity and issues signed bearer token.
        # How it works: Verifies password digest; raises AuthenticationError (401) on failure.
        user = self.user_repo.get_by_email(payload.email)
        if not user or not verify_password(payload.password, user.password_hash):
            logger.warning(f"Authentication failed for email '{payload.email}'.")
            raise AuthenticationError("Invalid email or password.")

        access_token = create_access_token(data={"sub": str(user.id), "role": user.role})
        logger.info({"event": "user_logged_in", "user_id": user.id})
        return Token(access_token=access_token, token_type="bearer")
