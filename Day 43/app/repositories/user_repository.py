# ==============================================================================
# Program    : UserRepository Data Access Layer (user_repository.py)
# Objective  : In-memory data store performing CRUD operations, pagination, and keyword search.
# Concept    : Repository Design Pattern (Day 43 requirement)
# Why Used   : Encapsulates direct data access logic cleanly away from business services.
# ==============================================================================

from typing import Any

class UserRepository:
    """Repository handling direct user data storage and query operations."""
    def __init__(self):
        self._users: list[dict[str, Any]] = [
            {"id": 1, "name": "Suraj Sawant", "email": "suraj@example.com", "age": 21},
            {"id": 2, "name": "Alex Mercer", "email": "alex@example.com", "age": 25},
            {"id": 3, "name": "John Doe", "email": "john@example.com", "age": 30},
            {"id": 4, "name": "Jane Smith", "email": "jane@example.com", "age": 28}
        ]
        self._next_id = 5

    def get_all(self, skip: int = 0, limit: int = 10) -> list[dict[str, Any]]:
        return self._users[skip : skip + limit]

    def get_by_id(self, user_id: int) -> dict[str, Any] | None:
        for u in self._users:
            if u["id"] == user_id:
                return u
        return None

    def get_by_email(self, email: str) -> dict[str, Any] | None:
        for u in self._users:
            if u["email"].lower() == email.lower():
                return u
        return None

    def search_by_name(self, name: str) -> list[dict[str, Any]]:
        query = name.strip().lower()
        return [u for u in self._users if query in u["name"].lower()]

    def create(self, name: str, email: str, age: int) -> dict[str, Any]:
        new_user = {"id": self._next_id, "name": name, "email": email, "age": age}
        self._next_id += 1
        self._users.append(new_user)
        return new_user

    def update(self, user_id: int, name: str, email: str, age: int) -> dict[str, Any] | None:
        user = self.get_by_id(user_id)
        if not user:
            return None
        user["name"] = name
        user["email"] = email
        user["age"] = age
        return user

    def delete(self, user_id: int) -> bool:
        user = self.get_by_id(user_id)
        if not user:
            return False
        self._users.remove(user)
        return True
