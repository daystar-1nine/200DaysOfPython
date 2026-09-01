# ==============================================================================
# Program    : Product REST API Router (products.py)
# Objective  : APIRouter for /products CRUD endpoints.
# Concept    : Product Resource Endpoints
# Why Used   : Exposes product management endpoints supporting categories and description fields.
# ==============================================================================

import os
import sys
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.database import get_db
from app.repositories.product_repository import ProductRepository
from app.schemas.product import ProductCreate, ProductUpdate, ProductPatch, ProductResponse
from app.services.product_service import ProductService

router = APIRouter(prefix="/products", tags=["Products"])

def get_product_service(db: Session = Depends(get_db)) -> ProductService:
    return ProductService(ProductRepository(db))

@router.get("", response_model=list[ProductResponse])
def list_products(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    service: ProductService = Depends(get_product_service)
):
    return service.list_products(skip=skip, limit=limit)

@router.get("/{product_id}", response_model=ProductResponse)
def get_product(product_id: int, service: ProductService = Depends(get_product_service)):
    return service.get_product(product_id=product_id)

@router.post("", response_model=ProductResponse, status_code=status.HTTP_201_CREATED)
def create_product(payload: ProductCreate, service: ProductService = Depends(get_product_service)):
    return service.create_product(payload=payload)

@router.put("/{product_id}", response_model=ProductResponse)
def replace_product(
    product_id: int,
    payload: ProductUpdate,
    service: ProductService = Depends(get_product_service)
):
    return service.replace_product(product_id=product_id, payload=payload)

@router.patch("/{product_id}", response_model=ProductResponse)
def patch_product(
    product_id: int,
    payload: ProductPatch,
    service: ProductService = Depends(get_product_service)
):
    return service.patch_product(product_id=product_id, payload=payload)

@router.delete("/{product_id}", status_code=status.HTTP_200_OK)
def delete_product(product_id: int, service: ProductService = Depends(get_product_service)):
    service.delete_product(product_id=product_id)
    return {"message": "Product deleted successfully", "id": product_id}
