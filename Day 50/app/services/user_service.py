"""
===============================================================================
DAY 50 — USER SERVICE (BUSINESS LOGIC LAYER)
===============================================================================
This module manages user profile retrieval and administrative user listing operations.
===============================================================================
"""

from typing import Sequence
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.exceptions import NotFoundError


class UserService:
    """Service executing operations on User accounts."""

    def __init__(self, user_repo: UserRepository) -> None:
        self.user_repo = user_repo

    def get_profile(self, user_id: int) -> User:
        """Retrieve user profile by ID."""
        # What is used: UserRepository get_by_id lookup with NotFoundError check.
        # Why it is used: Ensures requested user account exists.
        # How it works: Raises NotFoundError (404) if user record is missing.
        user = self.user_repo.get_by_id(user_id)
        if not user:
            raise NotFoundError(f"User with ID {user_id} not found.")
        return user

    def list_users(self) -> Sequence[User]:
        """List all users (Admin privilege)."""
        # What is used: UserRepository list_all invocation.
        # Why it is used: Provides complete listing of registered accounts for admin dashboard.
        # How it works: Delegates query execution to UserRepository.
        return self.user_repo.list_all()
