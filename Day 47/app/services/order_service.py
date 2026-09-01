# ==============================================================================
# Program    : Order Placement Business Service Layer (order_service.py)
# Objective  : Business logic for order placement, stock verification, stock deduction, and ownership authorization.
# Concept    : Atomic Database Transactions & Resource Ownership Checks
# Why Used   : Enforces stock deductions and ensures users can only access their own orders.
# ==============================================================================

from typing import List
from sqlalchemy.orm import Session
from app.models.user import User
from app.models.order import Order
from app.models.order_item import OrderItem
from app.repositories.user_repository import UserRepository
from app.repositories.product_repository import ProductRepository
from app.repositories.order_repository import OrderRepository
from app.schemas.order import OrderCreate
from app.exceptions import OrderNotFoundError, InsufficientStockError, PermissionDeniedError

class OrderService:
    def __init__(self, db: Session):
        self.db = db
        self.user_repo = UserRepository(db)
        self.product_repo = ProductRepository(db)
        self.order_repo = OrderRepository(db)

    def create_order(self, user_id: int, payload: OrderCreate) -> Order:
        """Execute atomic 8-step checkout transaction:

        1. Verify customer exists.
        2. Inspect stock for each line item.
        3. Deduct inventory stock.
        4. Create OrderItem junction records.
        5. Calculate total order price.
        6. Create Order record.
        7. Commit transaction.
        """
        user = self.user_repo.get_by_id(user_id)
        if not user:
            raise PermissionDeniedError("Cannot create order for unauthenticated user.")

        order_items_to_create = []
        calculated_total = 0.0

        try:
            for item_in in payload.items:
                product = self.product_repo.get_by_id(item_in.product_id)
                if not product:
                    raise OrderNotFoundError(item_in.product_id)

                if product.stock < item_in.quantity:
                    raise InsufficientStockError(
                        product_title=product.name,
                        requested=item_in.quantity,
                        available=product.stock
                    )

                # Deduct stock
                product.stock -= item_in.quantity
                line_price = product.price * item_in.quantity
                calculated_total += line_price

                order_item = OrderItem(
                    product_id=product.id,
                    quantity=item_in.quantity,
                    price=product.price
                )
                order_items_to_create.append(order_item)

            order = Order(
                user_id=user.id,
                total_amount=calculated_total,
                status="completed",
                items=order_items_to_create
            )

            created_order = self.order_repo.create(order)
            return created_order

        except Exception:
            self.db.rollback()
            raise

    def get_order_by_id(self, order_id: int, current_user: User) -> Order:
        """Fetch order by ID with ownership authorization check.

        - Normal users can ONLY access their own orders (HTTP 403 otherwise).
        - Admin users can access any order.
        """
        order = self.order_repo.get_by_id(order_id)
        if not order:
            raise OrderNotFoundError(order_id)

        if current_user.role != "admin" and order.user_id != current_user.id:
            raise PermissionDeniedError("You are not authorized to view this order.")

        return order

    def get_user_orders(self, user_id: int) -> List[Order]:
        """Fetch orders strictly belonging to user_id."""
        return self.order_repo.get_user_orders(user_id)

    def list_all_orders(self) -> List[Order]:
        """Fetch all orders across system (Admin use)."""
        return self.order_repo.list_all()
