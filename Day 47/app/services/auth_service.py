# ==============================================================================
# Program    : Authentication Business Service Layer (auth_service.py)
# Objective  : Business logic for user registration, password verification, and token issuance.
# Concept    : Separation of Authentication Business Rules from API Routers
# Why Used   : Hashes user passwords on register and validates credentials on login.
# ==============================================================================

import os
import sys
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse
from app.security import hash_password, verify_password, create_access_token
from app.exceptions import UserAlreadyExistsError, InvalidCredentialsError

class AuthService:
    def __init__(self, repository: UserRepository):
        self.repository = repository

    def register_user(self, payload: RegisterRequest) -> User:
        """Register new user account with hashed password."""
        existing = self.repository.get_by_email(payload.email)
        if existing:
            raise UserAlreadyExistsError(payload.email)

        hashed = hash_password(payload.password)
        role = payload.role if payload.role in ("user", "admin") else "user"

        user = User(
            name=payload.name,
            email=payload.email,
            age=payload.age,
            phone=payload.phone,
            password_hash=hashed,
            role=role
        )
        return self.repository.create(user)

    def authenticate_user(self, payload: LoginRequest) -> TokenResponse:
        """Authenticate user credentials and issue signed JWT access token."""
        user = self.repository.get_by_email(payload.email)
        if not user or not verify_password(payload.password, user.password_hash):
            raise InvalidCredentialsError()

        token_payload = {
            "sub": str(user.id),
            "email": user.email,
            "role": user.role
        }
        access_token = create_access_token(data=token_payload)
        return TokenResponse(access_token=access_token, token_type="bearer")
