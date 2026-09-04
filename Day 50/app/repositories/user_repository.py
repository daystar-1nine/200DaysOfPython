"""
===============================================================================
DAY 50 — USER REPOSITORY (DATA ACCESS LAYER)
===============================================================================
This module provides database abstraction for User entity persistence operations,
including query execution via SQLAlchemy 2.0 select statements.
===============================================================================
"""

from typing import List, Sequence
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.user import User


class UserRepository:
    """Repository handling SQL operations on the 'users' table."""

    def __init__(self, db: Session) -> None:
        # What is used: Dependency-injected SQLAlchemy Session.
        # Why it is used: Encapsulates database session access within repository layer.
        # How it works: Assigns db instance attribute.
        self.db = db

    def get_by_id(self, user_id: int) -> User | None:
        """Fetch a single user by primary key ID."""
        # What is used: SQLAlchemy 2.0 select query with scalar_one_or_none.
        # Why it is used: Safely executes SELECT WHERE id = user_id.
        # How it works: Compiles select(User) and returns ORM object or None.
        stmt = select(User).where(User.id == user_id)
        return self.db.execute(stmt).scalar_one_or_none()

    def get_by_email(self, email: str) -> User | None:
        """Fetch a single user by email address."""
        # What is used: Select statement filtering by email.
        # Why it is used: Checks user existence for registration or login.
        # How it works: Filters User.email == email.lower().
        stmt = select(User).where(User.email == email.lower())
        return self.db.execute(stmt).scalar_one_or_none()

    def create(self, user: User) -> User:
        """Persist a new User entity."""
        # What is used: Session add, commit, and refresh operations.
        # Why it is used: Inserts user row into table and loads auto-generated ID/timestamps.
        # How it works: Adds user to db, commits transaction, and refreshes instance.
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def list_all(self) -> Sequence[User]:
        """Fetch all registered users (admin privilege)."""
        # What is used: Select query returning all User instances.
        # Why it is used: Provides data list for admin management endpoint.
        # How it works: Executes select(User) order_by(User.id).
        stmt = select(User).order_by(User.id.asc())
        return self.db.execute(stmt).scalars().all()
