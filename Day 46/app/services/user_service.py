# ==============================================================================
# Program    : UserService (user_service.py)
# Objective  : UserService delegating to UserRepository and enforcing unique email rules.
# Concept    : Layered Business Architecture
# Why Used   : Manages user entity operations including phone numbers.
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
from app.schemas.user import UserCreate

class UserService:
    def __init__(self, repository: UserRepository):
        self.repository = repository

    def list_users(self, skip: int = 0, limit: int = 10) -> Sequence[User]:
        return self.repository.get_all(skip=skip, limit=limit)

    def get_user(self, user_id: int) -> User:
        user = self.repository.get_by_id(user_id)
        if not user:
            raise UserNotFoundError(user_id)
        return user

    def get_user_with_orders(self, user_id: int) -> User:
        user = self.repository.get_by_id_with_orders(user_id)
        if not user:
            raise UserNotFoundError(user_id)
        return user

    def create_user(self, payload: UserCreate) -> User:
        existing = self.repository.get_by_email(payload.email)
        if existing:
            raise UserAlreadyExistsError(payload.email)
        return self.repository.create(name=payload.name, email=payload.email, phone=payload.phone)
