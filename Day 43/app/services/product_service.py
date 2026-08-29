# ==============================================================================
# Program    : ProductService Business Logic Layer (product_service.py)
# Objective  : ProductService executing business operations using ProductRepository.
# Concept    : Layered Architecture
# Why Used   : Encapsulates product business logic.
# ==============================================================================

import os
import sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.exceptions import ProductNotFoundError
from app.models.product import ProductCreate
from app.repositories.product_repository import ProductRepository

class ProductService:
    """Service class handling product business logic."""
    def __init__(self, repository: ProductRepository):
        self.repository = repository

    def list_products(self) -> list[dict]:
        return self.repository.get_all()

    def get_product(self, product_id: int) -> dict:
        product = self.repository.get_by_id(product_id)
        if not product:
            raise ProductNotFoundError(product_id)
        return product

    def create_product(self, payload: ProductCreate) -> dict:
        return self.repository.create(title=payload.title, price=payload.price)
