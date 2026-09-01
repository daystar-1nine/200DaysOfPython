# ==============================================================================
# Program    : Products Catalog APIRouter (products.py)
# Objective  : Implement public catalog reading and admin-only product creation/update/deletion endpoints.
# Concept    : Role-Based Authorization on Catalog Mutating Endpoints
# Why Used   : Allows public browsing while restricting catalog modifications strictly to admin users.
# ==============================================================================

import os
import sys
from typing import List
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.database import get_db
from app.models.user import User
from app.dependencies.auth import require_admin
from app.schemas.product import ProductCreate, ProductUpdate, ProductResponse
from app.repositories.product_repository import ProductRepository
from app.services.product_service import ProductService

router = APIRouter(prefix="/products", tags=["Products Catalog"])

def get_product_service(db: Session = Depends(get_db)) -> ProductService:
    return ProductService(ProductRepository(db))

@router.get("", response_model=List[ProductResponse])
def list_products(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    service: ProductService = Depends(get_product_service)
):
    """List products catalog items (Public access)."""
    return service.list_products(skip=skip, limit=limit)

@router.get("/{product_id}", response_model=ProductResponse)
def get_product(
    product_id: int,
    service: ProductService = Depends(get_product_service)
):
    """Get single product details by ID (Public access)."""
    return service.get_product(product_id)

@router.post("", response_model=ProductResponse, status_code=status.HTTP_201_CREATED)
def create_product(
    payload: ProductCreate,
    admin_user: User = Depends(require_admin),
    service: ProductService = Depends(get_product_service)
):
    """Create new product catalog item (Admin role required; HTTP 403 Forbidden for non-admins)."""
    return service.create_product(payload)

@router.patch("/{product_id}", response_model=ProductResponse)
def patch_product(
    product_id: int,
    payload: ProductUpdate,
    admin_user: User = Depends(require_admin),
    service: ProductService = Depends(get_product_service)
):
    """Partially update product details (Admin role required; HTTP 403 Forbidden for non-admins)."""
    return service.patch_product(product_id, payload)

@router.delete("/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_product(
    product_id: int,
    admin_user: User = Depends(require_admin),
    service: ProductService = Depends(get_product_service)
):
    """Delete product from catalog (Admin role required; HTTP 403 Forbidden for non-admins)."""
    service.delete_product(product_id)
