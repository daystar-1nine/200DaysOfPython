# ==============================================================================
# Program    : Order REST API Router (orders.py)
# Objective  : APIRouter for POST /orders, GET /orders, GET /orders/{id}.
# Concept    : Order Resource APIRouter
# Why Used   : Exposes order placement and order details retrieval endpoints.
# ==============================================================================

import os
import sys
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.database import get_db
from app.schemas.order import OrderCreate, OrderResponse
from app.services.order_service import OrderService

router = APIRouter(prefix="/orders", tags=["Orders"])

def get_order_service(db: Session = Depends(get_db)) -> OrderService:
    return OrderService(db)

@router.get("", response_model=list[OrderResponse])
def list_orders(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    service: OrderService = Depends(get_order_service)
):
    return service.list_orders(skip=skip, limit=limit)

@router.get("/{order_id}", response_model=OrderResponse)
def get_order(order_id: int, service: OrderService = Depends(get_order_service)):
    return service.get_order(order_id=order_id)

@router.post("", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
def create_order(payload: OrderCreate, service: OrderService = Depends(get_order_service)):
    return service.create_order(payload=payload)
