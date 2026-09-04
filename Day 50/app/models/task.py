"""
===============================================================================
DAY 50 — TASK ORM MODEL DEFINITION
===============================================================================
This module defines the Task SQLAlchemy 2.0 declarative ORM model mapping to
the 'tasks' database table with status/priority enums and foreign key to user.
===============================================================================
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from sqlalchemy import String, Text, ForeignKey, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base

if TYPE_CHECKING:
    from app.models.user import User


class Task(Base):
    """Task ORM entity representing user task items."""

    __tablename__ = "tasks"

    # What is used: Primary key column declaration.
    # Why it is used: Uniquely identifies task entity records.
    # How it works: Auto-increment primary key index.
    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # What is used: Task attribute columns (title, description, status, priority, due_date).
    # Why it is used: Stores task details, status progress state, and priority rank.
    # How it works: Defaults status to 'TODO' and priority to 'MEDIUM'.
    title: Mapped[str] = mapped_column(String(200), index=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="TODO", index=True, nullable=False)
    priority: Mapped[str] = mapped_column(String(20), default="MEDIUM", index=True, nullable=False)
    due_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # What is used: Foreign key column referencing users.id.
    # Why it is used: Establishes referential integrity link to owning User entity.
    # How it works: Stores integer user_id matching users.id table primary key.
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # What is used: Timestamp audit columns for creation and modification.
    # Why it is used: Tracks when task was created and last updated.
    # How it works: Defaults created_at and updated_at to UTC timestamp.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # What is used: Many-to-one relationship mapping to User entity.
    # Why it is used: Allows Task entity instance to access owner User entity.
    # How it works: Configures relationship back_populates="tasks".
    owner: Mapped["User"] = relationship("User", back_populates="tasks")
