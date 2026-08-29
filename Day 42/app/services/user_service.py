# ==============================================================================
# Program    : User Business Service (user_service.py)
# Objective  : Business logic and in-memory list storage for User CRUD operations, pagination, and search.
# Concept    : Service Layer Architecture (Day 31 OOP)
# Why Used   : Decouples REST API web routes from underlying data persistence logic.
# ==============================================================================

import os
import sys
from typing import Any

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.exceptions import UserNotFoundError, UserAlreadyExistsError
from app.models.user import UserCreate, UserUpdate, UserPatch

class UserService:
    """Service handling User business logic and in-memory data storage."""
    def __init__(self):
        # Initial seed dataset
        self._users: list[dict[str, Any]] = [
            {"id": 1, "name": "Suraj Sawant", "email": "suraj@example.com", "age": 21},
            {"id": 2, "name": "Alex Mercer", "email": "alex@example.com", "age": 25},
            {"id": 3, "name": "John Doe", "email": "john@example.com", "age": 30},
            {"id": 4, "name": "Jane Smith", "email": "jane@example.com", "age": 28}
        ]
        self._next_id = 5

    def list_users(self, skip: int = 0, limit: int = 10) -> list[dict[str, Any]]:
        """Returns paginated slice of user list using skip and limit parameters."""
        return self._users[skip : skip + limit]

    def get_user(self, user_id: int) -> dict[str, Any]:
        """Fetches single user dict by ID or raises UserNotFoundError."""
        for u in self._users:
            if u["id"] == user_id:
                return u
        raise UserNotFoundError(user_id)

    def search_users(self, name: str) -> list[dict[str, Any]]:
        """Filters users whose names contain case-insensitive keyword."""
        query = name.strip().lower()
        return [u for u in self._users if query in u["name"].lower()]

    def create_user(self, payload: UserCreate) -> dict[str, Any]:
        """Creates new user record after checking email uniqueness."""
        for u in self._users:
            if u["email"].lower() == payload.email.lower():
                raise UserAlreadyExistsError(payload.email)

        new_user = {
            "id": self._next_id,
            "name": payload.name,
            "email": payload.email,
            "age": payload.age
        }
        self._next_id += 1
        self._users.append(new_user)
        return new_user

    def replace_user(self, user_id: int, payload: UserUpdate) -> dict[str, Any]:
        """Replaces entire user object (PUT semantics)."""
        user = self.get_user(user_id)
        user["name"] = payload.name
        user["email"] = payload.email
        user["age"] = payload.age
        return user

    def patch_user(self, user_id: int, payload: UserPatch) -> dict[str, Any]:
        """Partially modifies specified fields of user object (PATCH semantics)."""
        user = self.get_user(user_id)
        if payload.name is not None:
            user["name"] = payload.name
        if payload.email is not None:
            user["email"] = payload.email
        if payload.age is not None:
            user["age"] = payload.age
        return user

    def delete_user(self, user_id: int) -> dict[str, Any]:
        """Deletes user object by ID."""
        user = self.get_user(user_id)
        self._users.remove(user)
        return user
