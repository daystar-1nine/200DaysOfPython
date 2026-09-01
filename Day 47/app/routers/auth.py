# ==============================================================================
# Program    : Authentication APIRouter (auth.py)
# Objective  : Define POST /auth/register and POST /auth/login endpoints.
# Concept    : Registration & Login Authentication Controller
# Why Used   : Handles user registration and issues Bearer access tokens.
# ==============================================================================

import os
import sys
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.database import get_db
from app.repositories.user_repository import UserRepository
from app.services.auth_service import AuthService
from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse
from app.schemas.user import UserResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])

def get_auth_service(db: Session = Depends(get_db)) -> AuthService:
    return AuthService(UserRepository(db))

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, service: AuthService = Depends(get_auth_service)):
    """Register new user account with hashed password."""
    return service.register_user(payload)

@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, service: AuthService = Depends(get_auth_service)):
    """Authenticate user credentials and return signed JWT access token."""
    return service.authenticate_user(payload)
