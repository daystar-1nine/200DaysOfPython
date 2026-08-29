# ==============================================================================
# Program    : User REST API Router (users.py)
# Objective  : Define APIRouter routes for /users CRUD endpoints, pagination, and search.
# Concept    : Modular Web API Routing (FastAPI APIRouter)
# Why Used   : Encapsulates user HTTP endpoints cleanly away from main.py.
# ==============================================================================

import os
import sys
from fastapi import APIRouter, Query, status

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.models.user import UserCreate, UserUpdate, UserPatch, UserResponse
from app.services.user_service import UserService

router = APIRouter(prefix="/users", tags=["Users"])
user_service = UserService()

# What is used : GET /users/search (Placed before /users/{user_id} to avoid route collision)
# Why it is used: Searches users matching name query parameter
@router.get("/search", response_model=list[UserResponse])
def search_users(name: str = Query(..., min_length=1, description="Name keyword to search")):
    return user_service.search_users(name=name)

# What is used : GET /users (Query Parameters Pagination)
# Why it is used: Slices user collection using skip and limit parameters
@router.get("", response_model=list[UserResponse])
def list_users(
    skip: int = Query(0, ge=0, description="Records to skip"),
    limit: int = Query(10, ge=1, le=100, description="Max records to return")
):
    return user_service.list_users(skip=skip, limit=limit)

# What is used : GET /users/{user_id} (Path Parameter)
# Why it is used: Fetches single user entity by unique integer ID
@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: int):
    return user_service.get_user(user_id=user_id)

# What is used : POST /users (Request Body Validation)
# Why it is used: Creates new user entity with Pydantic validation
@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(payload: UserCreate):
    return user_service.create_user(payload=payload)

# What is used : PUT /users/{user_id} (Full Replacement)
# Why it is used: Replaces entire existing user record
@router.put("/{user_id}", response_model=UserResponse)
def replace_user(user_id: int, payload: UserUpdate):
    return user_service.replace_user(user_id=user_id, payload=payload)

# What is used : PATCH /users/{user_id} (Partial Update)
# Why it is used: Updates specified attributes of user record
@router.patch("/{user_id}", response_model=UserResponse)
def patch_user(user_id: int, payload: UserPatch):
    return user_service.patch_user(user_id=user_id, payload=payload)

# What is used : DELETE /users/{user_id} (Resource Deletion)
# Why it is used: Deletes user record by integer ID
@router.delete("/{user_id}", status_code=status.HTTP_200_OK)
def delete_user(user_id: int):
    user_service.delete_user(user_id=user_id)
    return {"message": "User deleted successfully", "id": user_id}
