# ==============================================================================
# Program    : User ORM Model (user.py)
# Objective  : Define User model with phone, created_at, and 1:N relationship to Order models.
# Concept    : SQLAlchemy Declarative Model with Alembic Migration Support
# Why Used   : Represents 'users' table in database with index on email column.
# ==============================================================================

import os
import sys
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import String, Integer, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.database import Base

class User(Base):
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    phone: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=True)

    # 1:N relationship to Order
    orders: Mapped[List["Order"]] = relationship(
        "Order",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} name='{self.name}' email='{self.email}' phone='{self.phone}'>"
