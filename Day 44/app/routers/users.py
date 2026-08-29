# ==============================================================================
# Program    : User REST API Router (users.py)
# Objective  : APIRouter for /users CRUD endpoints, pagination, and database ILIKE search.
# Concept    : FastAPI APIRouter & Database Session Injection
# Why Used   : Connects HTTP route endpoints to database-backed UserService and UserRepository.
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
from app.schemas.user import UserCreate, UserUpdate, UserPatch, UserResponse
from app.services.user_service import UserService

router = APIRouter(prefix="/users", tags=["Users"])

def get_user_service(db: Session = Depends(get_db)) -> UserService:
    """Dependency provider injecting database Session into UserRepository and UserService."""
    repo = UserRepository(db)
    return UserService(repo)

# What is used : GET /users/search (Database Search Query)
# Why it is used: Executes ILIKE SQL search on PostgreSQL/SQLite database engine
@router.get("/search", response_model=list[UserResponse])
def search_users(
    name: str = Query(..., min_length=1, description="Name keyword to search"),
    service: UserService = Depends(get_user_service)
):
    return service.search_users(name=name)

# What is used : GET /users (Paginated List)
# Why it is used: Returns paginated slice of database user records
@router.get("", response_model=list[UserResponse])
def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    service: UserService = Depends(get_user_service)
):
    return service.list_users(skip=skip, limit=limit)

# What is used : GET /users/{user_id}
# Why it is used: Fetches single database user record by ID
@router.get("/{user_id}", response_model=UserResponse)
def get_user(
    user_id: int,
    service: UserService = Depends(get_user_service)
):
    return service.get_user(user_id=user_id)

# What is used : POST /users
# Why it is used: Persists new user record into database
@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    payload: UserCreate,
    service: UserService = Depends(get_user_service)
):
    return service.create_user(payload=payload)

# What is used : PUT /users/{user_id}
# Why it is used: Replaces user record in database
@router.put("/{user_id}", response_model=UserResponse)
def replace_user(
    user_id: int,
    payload: UserUpdate,
    service: UserService = Depends(get_user_service)
):
    return service.replace_user(user_id=user_id, payload=payload)

# What is used : PATCH /users/{user_id}
# Why it is used: Updates specified attributes of database user record
@router.patch("/{user_id}", response_model=UserResponse)
def patch_user(
    user_id: int,
    payload: UserPatch,
    service: UserService = Depends(get_user_service)
):
    return service.patch_user(user_id=user_id, payload=payload)

# What is used : DELETE /users/{user_id}
# Why it is used: Deletes user record from database
@router.delete("/{user_id}", status_code=status.HTTP_200_OK)
def delete_user(
    user_id: int,
    service: UserService = Depends(get_user_service)
):
    service.delete_user(user_id=user_id)
    return {"message": "User deleted successfully", "id": user_id}
