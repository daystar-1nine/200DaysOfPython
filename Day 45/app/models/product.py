# ==============================================================================
# Program    : Product ORM Model (product.py)
# Objective  : Define Product model with price and inventory stock count.
# Concept    : SQLAlchemy 2.0 Declarative Model
# Why Used   : Represents 'products' table in database.
# ==============================================================================

import os
import sys
from typing import List
from sqlalchemy import String, Integer, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.database import Base

class Product(Base):
    __tablename__ = "products"
    __table_args__ = {'extend_existing': True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    stock: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # 1:N relationship to OrderItem
    order_items: Mapped[List["OrderItem"]] = relationship("OrderItem", back_populates="product")

    def __repr__(self) -> str:
        return f"<Product id={self.id} name='{self.name}' price={self.price} stock={self.stock}>"
