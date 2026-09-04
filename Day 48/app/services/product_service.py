# ==============================================================================
# Program    : Product Service Layer (product_service.py)
# Objective  : Business logic for managing product catalog items.
# Concept    : Service Layer Pattern
# Why Used   : Provides catalog management methods for product creation, updates, and deletions.
# ==============================================================================

from typing import List
from sqlalchemy.orm import Session
from app.models.product import Product
from app.schemas.product import ProductCreate, ProductUpdate
from app.repositories.product_repository import ProductRepository
from app.exceptions import ProductNotFoundError

class ProductService:
    def __init__(self, db: Session):
        self.repo = ProductRepository(db)

    def get_product(self, product_id: int) -> Product:
        product = self.repo.get_by_id(product_id)
        if not product:
            raise ProductNotFoundError(product_id)
        return product

    def list_products(self, skip: int = 0, limit: int = 100) -> List[Product]:
        return self.repo.list_all(skip=skip, limit=limit)

    def create_product(self, req: ProductCreate) -> Product:
        product = Product(
            name=req.name,
            price=req.price,
            stock=req.stock,
            description=req.description,
            category=req.category
        )
        return self.repo.create(product)

    def update_product(self, product_id: int, req: ProductUpdate) -> Product:
        product = self.get_product(product_id)
        update_data = req.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(product, key, value)
        return self.repo.update(product)

    def delete_product(self, product_id: int) -> None:
        product = self.get_product(product_id)
        self.repo.delete(product)
