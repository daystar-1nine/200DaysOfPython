# ==============================================================================
# Program    : Product Business Service Layer (product_service.py)
# Objective  : Provide product catalog retrieval and admin-only creation/update/deletion methods.
# Concept    : Service Layer Architecture
# Why Used   : Enforces product availability and catalog management business rules.
# ==============================================================================

from typing import List
from app.models.product import Product
from app.repositories.product_repository import ProductRepository
from app.schemas.product import ProductCreate, ProductUpdate
from app.exceptions import ProductNotFoundError

class ProductService:
    def __init__(self, repository: ProductRepository):
        self.repository = repository

    def get_product(self, product_id: int) -> Product:
        product = self.repository.get_by_id(product_id)
        if not product:
            raise ProductNotFoundError(product_id)
        return product

    def list_products(self, skip: int = 0, limit: int = 100) -> List[Product]:
        return self.repository.list_all(skip=skip, limit=limit)

    def create_product(self, payload: ProductCreate) -> Product:
        product = Product(
            name=payload.name,
            price=payload.price,
            stock=payload.stock,
            description=payload.description,
            category=payload.category
        )
        return self.repository.create(product)

    def patch_product(self, product_id: int, payload: ProductUpdate) -> Product:
        product = self.get_product(product_id)
        if payload.name is not None:
            product.name = payload.name
        if payload.price is not None:
            product.price = payload.price
        if payload.stock is not None:
            product.stock = payload.stock
        if payload.description is not None:
            product.description = payload.description
        if payload.category is not None:
            product.category = payload.category

        return self.repository.update(product)

    def delete_product(self, product_id: int) -> None:
        product = self.get_product(product_id)
        self.repository.delete(product)
