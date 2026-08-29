# ==============================================================================
# Program    : User REST API Router (users.py)
# Objective  : APIRouter for /users CRUD, pagination, search, and protected /profile endpoint.
# Concept    : FastAPI Routers & Dependency Injection (Day 43 requirement)
# Why Used   : Uses Depends(get_user_service) and Depends(get_current_user) for clean route design.
# ==============================================================================

import os
import sys
from fastapi import APIRouter, Depends, Query, status

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.dependencies.auth import get_current_user
from app.dependencies.providers import get_user_service
from app.models.user import UserCreate, UserUpdate, UserPatch, UserResponse, UserProfileResponse
from app.services.user_service import UserService

router = APIRouter(tags=["Users"])

# What is used : GET /profile Protected Endpoint (Day 43 requirement)
# Why it is used: Demonstrates dependency-based authentication via Depends(get_current_user)
@router.get("/profile", response_model=UserProfileResponse)
def get_user_profile(current_user: dict = Depends(get_current_user)):
    return current_user

# What is used : GET /users/search (Placed before /users/{id} to prevent path collision)
# Why it is used: Searches users matching query keyword
@router.get("/users/search", response_model=list[UserResponse])
def search_users(
    name: str = Query(..., min_length=1),
    service: UserService = Depends(get_user_service)
):
    return service.search_users(name=name)

# What is used : GET /users (Paginated List)
# Why it is used: Slices users using skip and limit parameters
@router.get("/users", response_model=list[UserResponse])
def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    service: UserService = Depends(get_user_service)
):
    return service.list_users(skip=skip, limit=limit)

# What is used : GET /users/{user_id}
# Why it is used: Fetches single user entity by ID
@router.get("/users/{user_id}", response_model=UserResponse)
def get_user(
    user_id: int,
    service: UserService = Depends(get_user_service)
):
    return service.get_user(user_id=user_id)

# What is used : POST /users
# Why it is used: Creates new user entity
@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    payload: UserCreate,
    service: UserService = Depends(get_user_service)
):
    return service.create_user(payload=payload)

# What is used : PUT /users/{user_id}
# Why it is used: Replaces entire user record
@router.put("/users/{user_id}", response_model=UserResponse)
def replace_user(
    user_id: int,
    payload: UserUpdate,
    service: UserService = Depends(get_user_service)
):
    return service.replace_user(user_id=user_id, payload=payload)

# What is used : PATCH /users/{user_id}
# Why it is used: Partially updates user record
@router.patch("/users/{user_id}", response_model=UserResponse)
def patch_user(
    user_id: int,
    payload: UserPatch,
    service: UserService = Depends(get_user_service)
):
    return service.patch_user(user_id=user_id, payload=payload)

# What is used : DELETE /users/{user_id}
# Why it is used: Deletes user record
@router.delete("/users/{user_id}", status_code=status.HTTP_200_OK)
def delete_user(
    user_id: int,
    service: UserService = Depends(get_user_service)
):
    service.delete_user(user_id=user_id)
    return {"message": "User deleted successfully", "id": user_id}
