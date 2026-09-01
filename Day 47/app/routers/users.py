# ==============================================================================
# Program    : User Profile & Admin APIRouter (users.py)
# Objective  : Implement GET /users/me, GET /users/me/orders, and GET /admin/users (Admin-only).
# Concept    : Protected Endpoints, Ownership Isolation & Role Authorization
# Why Used   : Restricts user access to personal profiles and protects admin tools with 403 checks.
# ==============================================================================

import os
import sys
from typing import List
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.database import get_db
from app.models.user import User
from app.dependencies.auth import get_current_user, require_admin
from app.schemas.user import UserResponse
from app.schemas.order import OrderResponse
from app.services.user_service import UserService
from app.services.order_service import OrderService
from app.repositories.user_repository import UserRepository

router = APIRouter(tags=["Users & Profile"])

def get_user_service(db: Session = Depends(get_db)) -> UserService:
    return UserService(UserRepository(db))

def get_order_service(db: Session = Depends(get_db)) -> OrderService:
    return OrderService(db)

@router.get("/users/me", response_model=UserResponse)
def read_current_user(current_user: User = Depends(get_current_user)):
    """Retrieve profile details of the authenticated current user."""
    return current_user

@router.get("/users/me/orders", response_model=List[OrderResponse])
def read_current_user_orders(
    current_user: User = Depends(get_current_user),
    service: OrderService = Depends(get_order_service)
):
    """Retrieve orders strictly belonging to the authenticated current user (Ownership Isolation)."""
    return service.get_user_orders(current_user.id)

@router.get("/admin/users", response_model=List[UserResponse])
def list_all_users_admin(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    admin_user: User = Depends(require_admin),
    service: UserService = Depends(get_user_service)
):
    """Retrieve all users in system (Admin role required; raises HTTP 403 Forbidden for non-admins)."""
    return service.list_all_users(skip=skip, limit=limit)
