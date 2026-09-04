"""
===============================================================================
DAY 50 — USERS ROUTER MODULE
===============================================================================
This module defines user profile endpoints (GET /users/me).
===============================================================================
"""

from fastapi import APIRouter, Depends, status
from app.models.user import User
from app.schemas.user import UserResponse
from app.dependencies.auth import get_current_user

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/me", response_model=UserResponse, status_code=status.HTTP_200_OK)
def get_current_user_profile(current_user: User = Depends(get_current_user)) -> UserResponse:
    """Retrieve currently authenticated user profile details."""
    # What is used: get_current_user dependency injection.
    # Why it is used: Returns authenticated user payload.
    # How it works: Serializes current_user model instance to UserResponse DTO.
    return UserResponse.model_validate(current_user)
