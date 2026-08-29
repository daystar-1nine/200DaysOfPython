# ==============================================================================
# Program    : ProductRepository (product_repository.py)
# Objective  : Data access layer for Product models with inventory stock tracking.
# Concept    : Database Persistence
# Why Used   : Provides database queries for product management and inventory updates.
# ==============================================================================

import os
import sys
from typing import Sequence
from sqlalchemy import select
from sqlalchemy.orm import Session

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.models.product import Product

class ProductRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all(self, skip: int = 0, limit: int = 10) -> Sequence[Product]:
        stmt = select(Product).offset(skip).limit(limit)
        return self.db.scalars(stmt).all()

    def get_by_id(self, product_id: int) -> Product | None:
        return self.db.get(Product, product_id)

    def create(self, name: str, price: float, stock: int) -> Product:
        product = Product(name=name, price=price, stock=stock)
        self.db.add(product)
        self.db.commit()
        self.db.refresh(product)
        return product

    def update(self, product_id: int, name: str, price: float, stock: int) -> Product | None:
        product = self.get_by_id(product_id)
        if not product:
            return None
        product.name = name
        product.price = price
        product.stock = stock
        self.db.commit()
        self.db.refresh(product)
        return product

    def delete(self, product_id: int) -> bool:
        product = self.get_by_id(product_id)
        if not product:
            return False
        self.db.delete(product)
        self.db.commit()
        return True
