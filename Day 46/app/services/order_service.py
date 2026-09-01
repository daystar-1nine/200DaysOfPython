# ==============================================================================
# Program    : OrderService Transactional Business Engine (order_service.py)
# Objective  : Execute atomic 8-step transactional order placement with inventory stock deductions.
# Concept    : Database Transactions & Rollback Protection
# Why Used   : Ensures order placement is completely atomic—rolling back on out-of-stock or invalid entities.
# ==============================================================================

import os
import sys
from typing import Sequence
from sqlalchemy.orm import Session

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.exceptions import UserNotFoundError, ProductNotFoundError, OrderNotFoundError, InsufficientStockError
from app.models.order import Order
from app.repositories.order_repository import OrderRepository
from app.repositories.user_repository import UserRepository
from app.repositories.product_repository import ProductRepository
from app.schemas.order import OrderCreate

class OrderService:
    def __init__(self, db: Session):
        self.db = db
        self.order_repo = OrderRepository(db)
        self.user_repo = UserRepository(db)
        self.product_repo = ProductRepository(db)

    def list_orders(self, skip: int = 0, limit: int = 10) -> Sequence[Order]:
        return self.order_repo.get_all(skip=skip, limit=limit)

    def get_order(self, order_id: int) -> Order:
        order = self.order_repo.get_by_id(order_id)
        if not order:
            raise OrderNotFoundError(order_id)
        return order

    def get_user_orders(self, user_id: int) -> Sequence[Order]:
        user = self.user_repo.get_by_id(user_id)
        if not user:
            raise UserNotFoundError(user_id)
        return self.order_repo.get_by_user_id(user_id)

    def create_order(self, payload: OrderCreate) -> Order:
        user = self.user_repo.get_by_id(payload.user_id)
        if not user:
            raise UserNotFoundError(payload.user_id)

        items_to_process = []
        total_amount = 0.0

        try:
            for item_in in payload.items:
                product = self.product_repo.get_by_id(item_in.product_id)
                if not product:
                    raise ProductNotFoundError(item_in.product_id)

                if product.stock < item_in.quantity:
                    raise InsufficientStockError(
                        product_title=product.name,
                        requested=item_in.quantity,
                        available=product.stock
                    )

                line_total = product.price * item_in.quantity
                total_amount += line_total
                items_to_process.append((product, item_in.quantity, product.price))

            order = self.order_repo.create(
                user_id=payload.user_id,
                total_amount=total_amount,
                status="COMPLETED"
            )

            for product, qty, price in items_to_process:
                self.order_repo.add_order_item(
                    order_id=order.id,
                    product_id=product.id,
                    quantity=qty,
                    price=price
                )
                product.stock -= qty

            self.db.commit()
            self.db.refresh(order)
            return self.get_order(order.id)

        except Exception as e:
            self.db.rollback()
            raise e
