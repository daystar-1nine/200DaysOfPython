# ==============================================================================
# Program    : Order ORM Model (order.py)
# Objective  : Define Order model with Foreign Key to User and 1:N to OrderItem.
# Concept    : Relational Model Mapping
# Why Used   : Stores customer order details, totals, and order line items.
# ==============================================================================

import os
import sys
from datetime import datetime, timezone
from typing import List
from sqlalchemy import String, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.database import Base

class Order(Base):
    __tablename__ = "orders"
    __table_args__ = {'extend_existing': True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="PENDING")
    total_amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="orders")
    items: Mapped[List["OrderItem"]] = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Order id={self.id} user_id={self.user_id} status='{self.status}' total={self.total_amount}>"
