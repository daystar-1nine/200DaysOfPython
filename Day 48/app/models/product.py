# ==============================================================================
# Program    : Product ORM Database Model (product.py)
# Objective  : Define Product table schema with name, price, stock, category, description, and created_at.
# Concept    : SQLAlchemy 2.0 Mapped ORM Model
# Why Used   : Represents products catalog items managed by admin users.
# ==============================================================================

from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from sqlalchemy import String, Float, Integer, Text, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base

if TYPE_CHECKING:
    from app.models.order_item import OrderItem

class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(150), index=True, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    stock: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    order_items: Mapped[List["OrderItem"]] = relationship("OrderItem", back_populates="product")

    def __repr__(self) -> str:
        return f"<Product(id={self.id}, name='{self.name}', price={self.price}, stock={self.stock})>"
