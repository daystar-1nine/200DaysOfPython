# ==============================================================================
# Program    : Orders Management APIRouter (orders.py)
# Objective  : Implement POST /orders (authenticated) and GET /orders/{id} (ownership/admin check).
# Concept    : Authenticated Order Checkout & Resource Ownership Checks
# Why Used   : Enforces stock deductions and isolates order inspection by current authenticated user.
# ==============================================================================

import os
import sys
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.database import get_db
from app.models.user import User
from app.dependencies.auth import get_current_user
from app.schemas.order import OrderCreate, OrderResponse
from app.services.order_service import OrderService

router = APIRouter(prefix="/orders", tags=["Orders Management"])

def get_order_service(db: Session = Depends(get_db)) -> OrderService:
    return OrderService(db)

@router.post("", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
def create_order(
    payload: OrderCreate,
    current_user: User = Depends(get_current_user),
    service: OrderService = Depends(get_order_service)
):
    """Place customer order for line items (Authenticated user required)."""
    return service.create_order(current_user.id, payload)

@router.get("/{order_id}", response_model=OrderResponse)
def get_order(
    order_id: int,
    current_user: User = Depends(get_current_user),
    service: OrderService = Depends(get_order_service)
):
    """Fetch order details by ID (Ownership check: owner or admin role required)."""
    return service.get_order_by_id(order_id, current_user)
