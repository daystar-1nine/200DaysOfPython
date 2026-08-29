# ==============================================================================
# Program    : Product REST API Router (products.py)
# Objective  : APIRouter for /products CRUD endpoints demonstrating multi-router architecture.
# Concept    : Modular APIRouter Component (Day 43 requirement)
# Why Used   : Demonstrates separating product routes cleanly from user routes.
# ==============================================================================

import os
import sys
from fastapi import APIRouter, Depends, status

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.dependencies.providers import get_product_service
from app.models.product import ProductCreate, ProductResponse
from app.services.product_service import ProductService

router = APIRouter(prefix="/products", tags=["Products"])

@router.get("", response_model=list[ProductResponse])
def list_products(service: ProductService = Depends(get_product_service)):
    return service.list_products()

@router.get("/{product_id}", response_model=ProductResponse)
def get_product(product_id: int, service: ProductService = Depends(get_product_service)):
    return service.get_product(product_id=product_id)

@router.post("", response_model=ProductResponse, status_code=status.HTTP_201_CREATED)
def create_product(payload: ProductCreate, service: ProductService = Depends(get_product_service)):
    return service.create_product(payload=payload)
