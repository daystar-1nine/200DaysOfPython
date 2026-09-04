# ==============================================================================
# Program    : User Service Layer (user_service.py)
# Objective  : Business logic for retrieving user profile information and admin user listings.
# Concept    : Service Layer Pattern
# Why Used   : Encapsulates user retrieval and security rule enforcement.
# ==============================================================================

from typing import List
from sqlalchemy.orm import Session
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.exceptions import UserNotFoundError

class UserService:
    def __init__(self, db: Session):
        self.repo = UserRepository(db)

    def get_user_by_id(self, user_id: int) -> User:
        user = self.repo.get_by_id(user_id)
        if not user:
            raise UserNotFoundError(user_id)
        return user

    def get_user_with_orders(self, user_id: int) -> User:
        user = self.repo.get_by_id_with_orders(user_id)
        if not user:
            raise UserNotFoundError(user_id)
        return user

    def list_all_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        return self.repo.list_all(skip=skip, limit=limit)
