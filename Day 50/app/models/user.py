"""
===============================================================================
DAY 50 — USER ORM MODEL DEFINITION
===============================================================================
This module defines the User SQLAlchemy 2.0 declarative ORM model mapping to
the 'users' database table, including relationships to Task entities.
===============================================================================
"""

from datetime import datetime, timezone
from typing import List, TYPE_CHECKING
from sqlalchemy import String, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base

if TYPE_CHECKING:
    from app.models.task import Task


class User(Base):
    """User ORM entity representing registered application accounts."""

    __tablename__ = "users"

    # What is used: Mapped primary key column declaration.
    # Why it is used: Uniquely identifies user entity records.
    # How it works: Generates auto-increment integer PK.
    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # What is used: User credentials and role column mapping.
    # Why it is used: Stores account identity details, password hash digest, and RBAC role.
    # How it works: Constrains email to unique index and sets default role='user'.
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="user", nullable=False)

    # What is used: Timestamp audit column with default factory.
    # Why it is used: Tracks account registration timestamp in UTC.
    # How it works: Sets default to current UTC timestamp on row creation.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # What is used: One-to-many relationship mapping to Task entity.
    # Why it is used: Enables navigation from User instance to owned Task instances.
    # How it works: Configures relationship back_populates="owner" with cascade deletion.
    tasks: Mapped[List["Task"]] = relationship(
        "Task",
        back_populates="owner",
        cascade="all, delete-orphan",
    )
