# ==============================================================================
# Program    : Product Catalog Management Router Module (products.py)
# Objective  : Public reading endpoints and admin-only mutation endpoints for products catalog.
# Concept    : Authorization Scoping & REST API Standards
# Why Used   : Public users can view products; only admins can create, update, or delete products.
# ==============================================================================

from typing import List
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.schemas.product import ProductCreate, ProductUpdate, ProductResponse
from app.dependencies.auth import require_admin
from app.services.product_service import ProductService

router = APIRouter(prefix="/products", tags=["Products"])

@router.get("", response_model=List[ProductResponse], status_code=status.HTTP_200_OK)
def list_products(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Public catalog endpoint listing available products."""
    service = ProductService(db)
    return service.list_products(skip=skip, limit=limit)

@router.get("/{product_id}", response_model=ProductResponse, status_code=status.HTTP_200_OK)
def get_product(product_id: int, db: Session = Depends(get_db)):
    """Public catalog endpoint retrieving details for a single product."""
    service = ProductService(db)
    return service.get_product(product_id)

@router.post("", response_model=ProductResponse, status_code=status.HTTP_201_CREATED)
def create_product(
    req: ProductCreate,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Admin-only endpoint creating a new product catalog item."""
    service = ProductService(db)
    return service.create_product(req)

@router.patch("/{product_id}", response_model=ProductResponse, status_code=status.HTTP_200_OK)
def update_product(
    product_id: int,
    req: ProductUpdate,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Admin-only endpoint updating an existing product catalog item."""
    service = ProductService(db)
    return service.update_product(product_id, req)

@router.delete("/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_product(
    product_id: int,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Admin-only endpoint deleting a product catalog item."""
    service = ProductService(db)
    service.delete_product(product_id)
    return None
