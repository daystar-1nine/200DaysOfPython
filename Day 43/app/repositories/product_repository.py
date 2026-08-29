# ==============================================================================
# Program    : ProductRepository Data Access Layer (product_repository.py)
# Objective  : In-memory storage for product CRUD operations.
# Concept    : Repository Pattern
# Why Used   : Provides data persistence interface for products router.
# ==============================================================================

from typing import Any

class ProductRepository:
    """Repository handling direct product storage operations."""
    def __init__(self):
        self._products: list[dict[str, Any]] = [
            {"id": 101, "title": "Mechanical Keyboard", "price": 79.99},
            {"id": 102, "title": "Wireless Gaming Mouse", "price": 49.99}
        ]
        self._next_id = 103

    def get_all(self) -> list[dict[str, Any]]:
        return self._products

    def get_by_id(self, product_id: int) -> dict[str, Any] | None:
        for p in self._products:
            if p["id"] == product_id:
                return p
        return None

    def create(self, title: str, price: float) -> dict[str, Any]:
        new_prod = {"id": self._next_id, "title": title, "price": price}
        self._next_id += 1
        self._products.append(new_prod)
        return new_prod
