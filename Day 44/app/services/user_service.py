# ==============================================================================
# Program    : UserService Business Logic Layer (user_service.py)
# Objective  : UserService delegating database operations to UserRepository and enforcing business rules.
# Concept    : Layered Architecture Business Service
# Why Used   : Enforces domain rules (unique email checks) before delegating to UserRepository.
# ==============================================================================

import os
import sys
from typing import Sequence

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.exceptions import UserNotFoundError, UserAlreadyExistsError
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.user import UserCreate, UserUpdate, UserPatch

class UserService:
    """Service class executing business rules and delegating to UserRepository."""
    def __init__(self, repository: UserRepository):
        self.repository = repository

    def list_users(self, skip: int = 0, limit: int = 10) -> Sequence[User]:
        return self.repository.get_all(skip=skip, limit=limit)

    def get_user(self, user_id: int) -> User:
        user = self.repository.get_by_id(user_id)
        if not user:
            raise UserNotFoundError(user_id)
        return user

    def search_users(self, name: str) -> Sequence[User]:
        return self.repository.search_by_name(name)

    def create_user(self, payload: UserCreate) -> User:
        existing = self.repository.get_by_email(payload.email)
        if existing:
            raise UserAlreadyExistsError(payload.email)
        return self.repository.create(name=payload.name, email=payload.email, age=payload.age)

    def replace_user(self, user_id: int, payload: UserUpdate) -> User:
        user = self.repository.update(user_id=user_id, name=payload.name, email=payload.email, age=payload.age)
        if not user:
            raise UserNotFoundError(user_id)
        return user

    def patch_user(self, user_id: int, payload: UserPatch) -> User:
        existing = self.get_user(user_id)
        name = payload.name if payload.name is not None else existing.name
        email = payload.email if payload.email is not None else existing.email
        age = payload.age if payload.age is not None else existing.age
        return self.repository.update(user_id=user_id, name=name, email=email, age=age)

    def delete_user(self, user_id: int) -> bool:
        success = self.repository.delete(user_id)
        if not success:
            raise UserNotFoundError(user_id)
        return True
