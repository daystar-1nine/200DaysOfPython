# ==============================================================================
# Program    : SQLAlchemy User ORM Model (user.py)
# Objective  : Define SQLAlchemy 2.0 ORM model representing 'users' database table.
# Concept    : DeclarativeBase, Mapped[T], mapped_column (Day 44 requirement)
# Why Used   : Maps Python User object instances directly to database table columns.
# ==============================================================================

import os
import sys
from sqlalchemy import String, Integer
from sqlalchemy.orm import Mapped, mapped_column

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.database import Base

class User(Base):
    """SQLAlchemy 2.0 ORM model representing users table in database."""
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}

    # What is used : Mapped[int] & mapped_column()
    # Why it is used: Modern SQLAlchemy 2.0 type-annotated column definitions
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)

    def __repr__(self) -> str:
        return f"<User id={self.id} name='{self.name}' email='{self.email}' age={self.age}>"
