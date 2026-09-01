# ==============================================================================
# Program    : User Business Service Layer (user_service.py)
# Objective  : Provide user profile retrieval and listing for admin/authenticated endpoints.
# Concept    : Service Layer Architecture
# Why Used   : Encapsulates user retrieval business rules.
# ==============================================================================

from typing import List
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.exceptions import UserNotFoundError

class UserService:
    def __init__(self, repository: UserRepository):
        self.repository = repository

    def get_user_by_id(self, user_id: int) -> User:
        user = self.repository.get_by_id(user_id)
        if not user:
            raise UserNotFoundError(user_id)
        return user

    def get_user_with_orders(self, user_id: int) -> User:
        user = self.repository.get_by_id_with_orders(user_id)
        if not user:
            raise UserNotFoundError(user_id)
        return user

    def list_all_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        return self.repository.list_all(skip=skip, limit=limit)
