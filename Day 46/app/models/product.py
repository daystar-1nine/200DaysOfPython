# ==============================================================================
# Program    : Product ORM Model (product.py)
# Objective  : Define Product model with description, category, and created_at columns.
# Concept    : Schema Evolution & Indexing Support
# Why Used   : Represents 'products' table with index on product name for fast catalog queries.
# ==============================================================================

import os
import sys
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import String, Integer, Float, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.database import Base

class Product(Base):
    __tablename__ = "products"
    __table_args__ = {'extend_existing': True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    stock: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=True)

    # 1:N relationship to OrderItem
    order_items: Mapped[List["OrderItem"]] = relationship("OrderItem", back_populates="product")

    def __repr__(self) -> str:
        return f"<Product id={self.id} name='{self.name}' price={self.price} category='{self.category}'>"
