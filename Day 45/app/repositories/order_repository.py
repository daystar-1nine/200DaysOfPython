# ==============================================================================
# Program    : OrderRepository (order_repository.py)
# Objective  : Data access layer for Order and OrderItem models with eager loading.
# Concept    : Order Persistence & Eager Loading (selectinload(Order.items))
# Why Used   : Retrieves orders along with line items in single query operations.
# ==============================================================================

import os
import sys
from typing import Sequence
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.models.order import Order
from app.models.order_item import OrderItem

class OrderRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all(self, skip: int = 0, limit: int = 10) -> Sequence[Order]:
        stmt = select(Order).options(selectinload(Order.items)).offset(skip).limit(limit)
        return self.db.scalars(stmt).all()

    def get_by_id(self, order_id: int) -> Order | None:
        stmt = select(Order).options(selectinload(Order.items)).where(Order.id == order_id)
        return self.db.scalars(stmt).first()

    def get_by_user_id(self, user_id: int) -> Sequence[Order]:
        stmt = select(Order).options(selectinload(Order.items)).where(Order.user_id == user_id)
        return self.db.scalars(stmt).all()

    def create(self, user_id: int, total_amount: float, status: str = "COMPLETED") -> Order:
        order = Order(user_id=user_id, total_amount=total_amount, status=status)
        self.db.add(order)
        self.db.flush()  # Flush to obtain order.id without committing transaction yet
        return order

    def add_order_item(self, order_id: int, product_id: int, quantity: int, price: float) -> OrderItem:
        item = OrderItem(order_id=order_id, product_id=product_id, quantity=quantity, price=price)
        self.db.add(item)
        return item
