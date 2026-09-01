# ==============================================================================
# Program    : UserRepository (user_repository.py)
# Objective  : Data access layer for User models using selectinload for eager loading.
# Concept    : Eager Loading & N+1 Query Prevention
# Why Used   : Fetches users and nested orders in an optimized 2-query batch operation.
# ==============================================================================

import os
import sys
from typing import Sequence, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.models.user import User

class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all(self, skip: int = 0, limit: int = 10) -> Sequence[User]:
        stmt = select(User).offset(skip).limit(limit)
        return self.db.scalars(stmt).all()

    def get_by_id_with_orders(self, user_id: int) -> User | None:
        stmt = select(User).options(selectinload(User.orders)).where(User.id == user_id)
        return self.db.scalars(stmt).first()

    def get_by_id(self, user_id: int) -> User | None:
        return self.db.get(User, user_id)

    def get_by_email(self, email: str) -> User | None:
        stmt = select(User).where(User.email.ilike(email.strip()))
        return self.db.scalars(stmt).first()

    def create(self, name: str, email: str, phone: Optional[str] = None) -> User:
        user = User(name=name, email=email, phone=phone)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user
