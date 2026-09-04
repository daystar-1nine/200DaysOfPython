# ==============================================================================
# Program    : Order Placement & Isolation Router Module (orders.py)
# Objective  : Route handlers for POST /orders and GET /orders/{id} with resource ownership checks.
# Concept    : Protected Endpoints & Resource Ownership Isolation
# Why Used   : Authenticated users place orders; non-admin users can ONLY read their own orders.
# ==============================================================================

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.schemas.order import OrderCreate, OrderResponse
from app.dependencies.auth import get_current_user
from app.services.order_service import OrderService

router = APIRouter(prefix="/orders", tags=["Orders"])

@router.post(
    "",
    response_model=OrderResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Place a new order",
    description="Authenticated endpoint placing a purchase order, calculating total, and deducting catalog inventory stock."
)
def create_order(
    req: OrderCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Authenticated endpoint placing a new order and deducting stock."""
    service = OrderService(db)
    return service.create_order(user_id=current_user.id, req=req)

@router.get(
    "/{order_id}",
    response_model=OrderResponse,
    status_code=status.HTTP_200_OK,
    summary="Get order details",
    description="Retrieves purchase order by ID. Enforces ownership isolation (403 Forbidden for non-owners)."
)
def get_order(
    order_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve an order by ID. Enforces ownership isolation (403 for non-owners)."""
    service = OrderService(db)
    return service.get_order_by_id(
        order_id=order_id,
        current_user_id=current_user.id,
        current_user_role=current_user.role
    )
