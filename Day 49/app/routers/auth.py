# ==============================================================================
# Program    : Authentication Router Module (auth.py)
# Objective  : Route handlers for user registration and credential authentication with OpenAPI docs.
# Concept    : APIRouter Modularization & OpenAPI Annotations
# Why Used   : Exposes POST /auth/register and POST /auth/login endpoints with Swagger UI documentation.
# ==============================================================================

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse
from app.schemas.user import UserResponse
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Registers a new user account with Argon2/Bcrypt hashed password credentials."
)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user account with hashed password credentials."""
    service = AuthService(db)
    return service.register_user(req)

@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Authenticate user credentials",
    description="Authenticates candidate email and password credentials and issues a signed JWT access token."
)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate user credentials and issue signed JWT access token."""
    service = AuthService(db)
    return service.authenticate_user(req)
