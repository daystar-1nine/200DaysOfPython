# ==============================================================================
# Program    : User Repository Data Access Layer (user_repository.py)
# Objective  : Provide CRUD operations for User entities with eager loading options.
# Concept    : Repository Pattern for Data Access Abstraction
# Why Used   : Isolates SQLAlchemy queries for users table.
# ==============================================================================

from typing import List, Optional
from sqlalchemy import select, func
from sqlalchemy.orm import Session, selectinload
from app.models.user import User

class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, user_id: int) -> Optional[User]:
        stmt = select(User).where(User.id == user_id)
        return self.db.scalars(stmt).first()

    def get_by_id_with_orders(self, user_id: int) -> Optional[User]:
        stmt = select(User).options(selectinload(User.orders)).where(User.id == user_id)
        return self.db.scalars(stmt).first()

    def get_by_email(self, email: str) -> Optional[User]:
        stmt = select(User).where(func.lower(User.email) == email.lower())
        return self.db.scalars(stmt).first()

    def list_all(self, skip: int = 0, limit: int = 100) -> List[User]:
        stmt = select(User).offset(skip).limit(limit)
        return list(self.db.scalars(stmt).all())

    def create(self, user: User) -> User:
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def update(self, user: User) -> User:
        self.db.commit()
        self.db.refresh(user)
        return user

    def delete(self, user: User) -> None:
        self.db.delete(user)
        self.db.commit()
