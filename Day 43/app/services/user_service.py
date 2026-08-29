# ==============================================================================
# Program    : UserService Business Logic Layer (user_service.py)
# Objective  : UserService delegating data access calls to UserRepository while enforcing validation rules.
# Concept    : Service Layer Architecture & Dependency Injection (Day 43 requirement)
# Why Used   : Enforces business logic rules (unique email checks, age validation) independently of HTTP routes.
# ==============================================================================

import os
import sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.exceptions import UserNotFoundError, UserAlreadyExistsError
from app.models.user import UserCreate, UserUpdate, UserPatch
from app.repositories.user_repository import UserRepository

class UserService:
    """Service class executing User business logic using injected UserRepository."""
    def __init__(self, repository: UserRepository):
        self.repository = repository

    def list_users(self, skip: int = 0, limit: int = 10) -> list[dict]:
        return self.repository.get_all(skip=skip, limit=limit)

    def get_user(self, user_id: int) -> dict:
        user = self.repository.get_by_id(user_id)
        if not user:
            raise UserNotFoundError(user_id)
        return user

    def search_users(self, name: str) -> list[dict]:
        return self.repository.search_by_name(name)

    def create_user(self, payload: UserCreate) -> dict:
        existing = self.repository.get_by_email(payload.email)
        if existing:
            raise UserAlreadyExistsError(payload.email)
        return self.repository.create(name=payload.name, email=payload.email, age=payload.age)

    def replace_user(self, user_id: int, payload: UserUpdate) -> dict:
        user = self.repository.update(user_id=user_id, name=payload.name, email=payload.email, age=payload.age)
        if not user:
            raise UserNotFoundError(user_id)
        return user

    def patch_user(self, user_id: int, payload: UserPatch) -> dict:
        existing = self.get_user(user_id)
        name = payload.name if payload.name is not None else existing["name"]
        email = payload.email if payload.email is not None else existing["email"]
        age = payload.age if payload.age is not None else existing["age"]
        return self.repository.update(user_id=user_id, name=name, email=email, age=age)

    def delete_user(self, user_id: int) -> bool:
        success = self.repository.delete(user_id)
        if not success:
            raise UserNotFoundError(user_id)
        return True
