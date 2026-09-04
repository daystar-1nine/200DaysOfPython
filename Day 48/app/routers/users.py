# ==============================================================================
# Program    : User Profile & Admin User Management Router (users.py)
# Objective  : Route handlers for /users/me, /users/me/orders, and /admin/users endpoints.
# Concept    : Protected Endpoints & Role-Based Authorization
# Why Used   : Demonstrates profile protection (401) and admin RBAC checks (403).
# ==============================================================================

from typing import List
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.schemas.user import UserResponse, UserWithOrdersResponse
from app.schemas.order import OrderResponse
from app.dependencies.auth import get_current_user, require_admin
from app.services.user_service import UserService
from app.services.order_service import OrderService

router = APIRouter(tags=["Users"])

@router.get("/users/me", response_model=UserResponse, status_code=status.HTTP_200_OK)
def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Retrieve authenticated user's profile metadata (excludes password_hash)."""
    return current_user

@router.get("/users/me/orders", response_model=List[OrderResponse], status_code=status.HTTP_200_OK)
def get_current_user_orders(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve authenticated user's order history."""
    order_service = OrderService(db)
    return order_service.list_user_orders(current_user.id)

@router.get("/admin/users", response_model=List[UserResponse], status_code=status.HTTP_200_OK)
def list_all_users_admin(
    skip: int = 0,
    limit: int = 100,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Admin-only endpoint listing all registered user accounts."""
    user_service = UserService(db)
    return user_service.list_all_users(skip=skip, limit=limit)
