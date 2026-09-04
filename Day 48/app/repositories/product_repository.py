# ==============================================================================
# Program    : Product Repository Data Access Layer (product_repository.py)
# Objective  : Provide CRUD operations for Product entities.
# Concept    : Repository Pattern for Data Access Abstraction
# Why Used   : Isolates SQLAlchemy queries for products catalog table.
# ==============================================================================

from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.product import Product

class ProductRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, product_id: int) -> Optional[Product]:
        stmt = select(Product).where(Product.id == product_id)
        return self.db.scalars(stmt).first()

    def list_all(self, skip: int = 0, limit: int = 100) -> List[Product]:
        stmt = select(Product).offset(skip).limit(limit)
        return list(self.db.scalars(stmt).all())

    def create(self, product: Product) -> Product:
        self.db.add(product)
        self.db.commit()
        self.db.refresh(product)
        return product

    def update(self, product: Product) -> Product:
        self.db.commit()
        self.db.refresh(product)
        return product

    def delete(self, product: Product) -> None:
        self.db.delete(product)
        self.db.commit()
