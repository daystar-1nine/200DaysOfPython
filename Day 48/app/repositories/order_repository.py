# ==============================================================================
# Program    : Order Repository Data Access Layer (order_repository.py)
# Objective  : Provide CRUD operations for Order entities with selectinload eager loading.
# Concept    : Repository Pattern & Eager Loading (selectinload)
# Why Used   : Eliminates N+1 query overhead by batch loading order items and product details.
# ==============================================================================

from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload
from app.models.order import Order
from app.models.order_item import OrderItem

class OrderRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, order_id: int) -> Optional[Order]:
        stmt = (
            select(Order)
            .options(selectinload(Order.items).selectinload(OrderItem.product))
            .where(Order.id == order_id)
        )
        return self.db.scalars(stmt).first()

    def get_user_orders(self, user_id: int) -> List[Order]:
        stmt = (
            select(Order)
            .options(selectinload(Order.items).selectinload(OrderItem.product))
            .where(Order.user_id == user_id)
            .order_by(Order.created_at.desc())
        )
        return list(self.db.scalars(stmt).all())

    def list_all(self, skip: int = 0, limit: int = 100) -> List[Order]:
        stmt = (
            select(Order)
            .options(selectinload(Order.items).selectinload(OrderItem.product))
            .offset(skip)
            .limit(limit)
            .order_by(Order.created_at.desc())
        )
        return list(self.db.scalars(stmt).all())

    def create(self, order: Order) -> Order:
        self.db.add(order)
        self.db.commit()
        self.db.refresh(order)
        return order
