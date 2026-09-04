# ==============================================================================
# Program    : Order Service Layer & Atomic Transactions (order_service.py)
# Objective  : Business logic for order placement, stock verification, and atomic transaction rollbacks with logging.
# Concept    : Atomic Transactions & Inventory Consistency
# Why Used   : Ensures orders and stock updates are processed atomically or rolled back completely on error.
# ==============================================================================

import logging
from typing import List
from sqlalchemy.orm import Session
from app.models.order import Order
from app.models.order_item import OrderItem
from app.schemas.order import OrderCreate
from app.repositories.order_repository import OrderRepository
from app.repositories.product_repository import ProductRepository
from app.exceptions import OrderNotFoundError, ProductNotFoundError, InsufficientStockError, AuthorizationError

logger = logging.getLogger("app.services.order_service")

class OrderService:
    def __init__(self, db: Session):
        self.db = db
        self.order_repo = OrderRepository(db)
        self.product_repo = ProductRepository(db)

    def create_order(self, user_id: int, req: OrderCreate) -> Order:
        """Process order creation atomically. Deducts stock per item or rolls back if stock insufficient."""
        logger.info(f"Order placement requested for user_id={user_id} with {len(req.items)} items")
        total_amount = 0.0
        order_items: List[OrderItem] = []

        try:
            for item_req in req.items:
                product = self.product_repo.get_by_id(item_req.product_id)
                if not product:
                    logger.warning(f"Order failed: product id={item_req.product_id} not found")
                    raise ProductNotFoundError(item_req.product_id)

                if product.stock < item_req.quantity:
                    logger.warning(
                        f"Order failed: insufficient stock for '{product.name}' (req={item_req.quantity}, avail={product.stock})"
                    )
                    raise InsufficientStockError(
                        product_title=product.name,
                        requested=item_req.quantity,
                        available=product.stock
                    )

                # Deduct inventory stock
                product.stock -= item_req.quantity
                item_total = product.price * item_req.quantity
                total_amount += item_total

                order_item = OrderItem(
                    product_id=product.id,
                    quantity=item_req.quantity,
                    price=product.price
                )
                order_items.append(order_item)

            new_order = Order(
                user_id=user_id,
                total_amount=round(total_amount, 2),
                status="completed",
                items=order_items
            )

            created = self.order_repo.create(new_order)
            logger.info(f"Order successfully placed with order_id={created.id}, total_amount={created.total_amount}")
            return created

        except Exception as e:
            logger.error(f"Order transaction aborted and rolled back due to error: {str(e)}")
            self.db.rollback()
            raise

    def get_order_by_id(self, order_id: int, current_user_id: int, current_user_role: str) -> Order:
        order = self.order_repo.get_by_id(order_id)
        if not order:
            logger.warning(f"Order not found with order_id={order_id}")
            raise OrderNotFoundError(order_id)

        # Enforce resource ownership isolation: non-admin users can ONLY read their own orders!
        if current_user_role != "admin" and order.user_id != current_user_id:
            logger.warning(f"Forbidden access: user_id={current_user_id} attempted to view order_id={order_id}")
            raise AuthorizationError("You do not have permission to view another user's order.")

        return order

    def list_user_orders(self, user_id: int) -> List[Order]:
        return self.order_repo.get_user_orders(user_id)

    def list_all_orders(self) -> List[Order]:
        return self.order_repo.list_all()
