# ==============================================================================
# Program    : User REST API Router (users.py)
# Objective  : APIRouter for /users CRUD and /users/{id}/orders nested endpoint.
# Concept    : Modular Routing & Nested Resource Responses
# Why Used   : Exposes user endpoints and nested user order collections.
# ==============================================================================

import os
import sys
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.database import get_db
from app.repositories.user_repository import UserRepository
from app.schemas.user import UserCreate, UserResponse, UserWithOrdersResponse
from app.schemas.order import OrderResponse
from app.services.user_service import UserService
from app.services.order_service import OrderService

router = APIRouter(prefix="/users", tags=["Users"])

def get_user_service(db: Session = Depends(get_db)) -> UserService:
    return UserService(UserRepository(db))

def get_order_service(db: Session = Depends(get_db)) -> OrderService:
    return OrderService(db)

@router.get("", response_model=list[UserResponse])
def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    service: UserService = Depends(get_user_service)
):
    return service.list_users(skip=skip, limit=limit)

@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: int, service: UserService = Depends(get_user_service)):
    return service.get_user(user_id=user_id)

@router.get("/{user_id}/with-orders", response_model=UserWithOrdersResponse)
def get_user_with_orders(user_id: int, service: UserService = Depends(get_user_service)):
    return service.get_user_with_orders(user_id=user_id)

@router.get("/{user_id}/orders", response_model=list[OrderResponse])
def get_user_orders(user_id: int, service: OrderService = Depends(get_order_service)):
    return service.get_user_orders(user_id=user_id)

@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(payload: UserCreate, service: UserService = Depends(get_user_service)):
    return service.create_user(payload=payload)
