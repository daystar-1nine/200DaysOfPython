# ==============================================================================
# Program    : User ORM Model (user.py)
# Objective  : Define User model with 1:N relationship to Order models.
# Concept    : SQLAlchemy 1:N Relationship (relationship back_populates)
# Why Used   : Represents 'users' table and provides ORM access to user orders.
# ==============================================================================

import os
import sys
from typing import List
from sqlalchemy import String, Integer
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

    # What is used : relationship("Order", back_populates="user")
    # Why it is used: Establishes 1:N relationship from User to Order models with cascading deletes
    orders: Mapped[List["Order"]] = relationship(
        "Order",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} name='{self.name}' email='{self.email}'>"
