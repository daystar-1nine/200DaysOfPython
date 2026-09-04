"""
===============================================================================
DAY 50 — AUTHENTICATION ROUTER MODULE
===============================================================================
This module defines endpoints for user registration (POST /auth/register) and
JWT login token generation (POST /auth/login).
===============================================================================
"""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.user import UserCreate, UserResponse, UserLogin, Token
from app.repositories.user_repository import UserRepository
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(payload: UserCreate, db: Session = Depends(get_db)) -> UserResponse:
    """Register a new user account."""
    # What is used: AuthService dependency injection and registration execution.
    # Why it is used: Validates payload and creates user account record.
    # How it works: Instantiates UserRepository and AuthService, returning UserResponse schema.
    user_repo = UserRepository(db)
    service = AuthService(user_repo)
    user = service.register(payload)
    return UserResponse.model_validate(user)


@router.post("/login", response_model=Token, status_code=status.HTTP_200_OK)
def login_user(payload: UserLogin, db: Session = Depends(get_db)) -> Token:
    """Authenticate user credentials and issue signed JWT access token."""
    # What is used: AuthService login execution.
    # Why it is used: Authenticates user credentials and issues bearer JWT token.
    # How it works: Verifies password digest and returns Token schema.
    user_repo = UserRepository(db)
    service = AuthService(user_repo)
    return service.login(payload)
