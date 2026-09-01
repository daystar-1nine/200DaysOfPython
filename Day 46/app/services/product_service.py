# ==============================================================================
# Program    : ProductService (product_service.py)
# Objective  : ProductService business logic layer supporting category, description, and stock.
# Concept    : Product Business Logic
# Why Used   : Enforces product creation, update, patch, and deletion validation rules.
# ==============================================================================

import os
import sys
from typing import Sequence

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.exceptions import ProductNotFoundError
from app.models.product import Product
from app.repositories.product_repository import ProductRepository
from app.schemas.product import ProductCreate, ProductUpdate, ProductPatch

class ProductService:
    def __init__(self, repository: ProductRepository):
        self.repository = repository

    def list_products(self, skip: int = 0, limit: int = 10) -> Sequence[Product]:
        return self.repository.get_all(skip=skip, limit=limit)

    def get_product(self, product_id: int) -> Product:
        product = self.repository.get_by_id(product_id)
        if not product:
            raise ProductNotFoundError(product_id)
        return product

    def create_product(self, payload: ProductCreate) -> Product:
        return self.repository.create(
            name=payload.name,
            price=payload.price,
            stock=payload.stock,
            description=payload.description,
            category=payload.category
        )

    def replace_product(self, product_id: int, payload: ProductUpdate) -> Product:
        product = self.repository.update(
            product_id,
            payload.name,
            payload.price,
            payload.stock,
            payload.description,
            payload.category
        )
        if not product:
            raise ProductNotFoundError(product_id)
        return product

    def patch_product(self, product_id: int, payload: ProductPatch) -> Product:
        existing = self.get_product(product_id)
        name = payload.name if payload.name is not None else existing.name
        price = payload.price if payload.price is not None else existing.price
        stock = payload.stock if payload.stock is not None else existing.stock
        description = payload.description if payload.description is not None else existing.description
        category = payload.category if payload.category is not None else existing.category
        return self.repository.update(product_id, name, price, stock, description, category)

    def delete_product(self, product_id: int) -> bool:
        success = self.repository.delete(product_id)
        if not success:
            raise ProductNotFoundError(product_id)
        return True
