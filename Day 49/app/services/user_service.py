# ==============================================================================
# Program    : User Service Layer (user_service.py)
# Objective  : Business logic for user profiles and admin user listings with logging.
# Concept    : Service Layer Pattern
# Why Used   : Encapsulates user retrieval and security checks.
# ==============================================================================

import logging
from typing import List
from sqlalchemy.orm import Session
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.exceptions import UserNotFoundError

logger = logging.getLogger("app.services.user_service")

class UserService:
    def __init__(self, db: Session):
        self.repo = UserRepository(db)

    def get_user_by_id(self, user_id: int) -> User:
        user = self.repo.get_by_id(user_id)
        if not user:
            logger.warning(f"User not found with id={user_id}")
            raise UserNotFoundError(user_id)
        return user

    def get_user_with_orders(self, user_id: int) -> User:
        user = self.repo.get_by_id_with_orders(user_id)
        if not user:
            logger.warning(f"User with orders not found with id={user_id}")
            raise UserNotFoundError(user_id)
        return user

    def list_all_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        logger.info(f"Listing all users (skip={skip}, limit={limit})")
        return self.repo.list_all(skip=skip, limit=limit)
