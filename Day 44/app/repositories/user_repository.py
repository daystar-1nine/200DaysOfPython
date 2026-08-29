# ==============================================================================
# Program    : UserRepository Database Data Access Layer (user_repository.py)
# Objective  : UserRepository executing database query operations via SQLAlchemy Session.
# Concept    : Repository Pattern + SQLAlchemy Session Queries (Day 44 requirement)
# Why Used   : Pushes query filtering and persistence operations down to PostgreSQL / SQLite database engine.
# ==============================================================================

import os
import sys
from typing import Sequence
from sqlalchemy import select
from sqlalchemy.orm import Session

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.models.user import User

class UserRepository:
    """Repository executing database persistence operations using SQLAlchemy Session."""
    def __init__(self, db: Session):
        self.db = db

    def get_all(self, skip: int = 0, limit: int = 10) -> Sequence[User]:
        """Fetches paginated slice of User models from database."""
        stmt = select(User).offset(skip).limit(limit)
        return self.db.scalars(stmt).all()

    def get_by_id(self, user_id: int) -> User | None:
        """Fetches single User model by primary key ID."""
        return self.db.get(User, user_id)

    def get_by_email(self, email: str) -> User | None:
        """Fetches single User model by unique email address."""
        stmt = select(User).where(User.email.ilike(email.strip()))
        return self.db.scalars(stmt).first()

    # What is used : Database SQL Filtering (ilike / icontains) (Day 44 requirement)
    # Why it is used: Pushes string search filtering directly into the Database engine rather than filtering in Python
    def search_by_name(self, name: str) -> Sequence[User]:
        """Executes SQL WHERE name ILIKE '%keyword%' search on database."""
        query_pattern = f"%{name.strip()}%"
        stmt = select(User).where(User.name.ilike(query_pattern))
        return self.db.scalars(stmt).all()

    def create(self, name: str, email: str, age: int) -> User:
        """Inserts new User record into database."""
        user = User(name=name, email=email, age=age)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def update(self, user_id: int, name: str, email: str, age: int) -> User | None:
        """Updates User record in database."""
        user = self.get_by_id(user_id)
        if not user:
            return None
        user.name = name
        user.email = email
        user.age = age
        self.db.commit()
        self.db.refresh(user)
        return user

    def delete(self, user_id: int) -> bool:
        """Deletes User record from database."""
        user = self.get_by_id(user_id)
        if not user:
            return False
        self.db.delete(user)
        self.db.commit()
        return True
