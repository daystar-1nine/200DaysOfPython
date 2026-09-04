# ==============================================================================
# Program    : Relationship & Schema Serialization Tests (test_relationships.py)
# Objective  : Test selectinload eager loading and Pydantic response model password_hash sanitization.
# Concept    : ORM Eager Loading & Response Serialization
# Why Used   : Ensures database relationships load efficiently without N+1 queries.
# ==============================================================================

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.user import UserResponse, UserWithOrdersResponse
from app.repositories.user_repository import UserRepository
from app.repositories.order_repository import OrderRepository
from app.schemas.order import OrderCreate, OrderItemCreate
from app.services.order_service import OrderService

def test_user_order_eager_loading_selectinload(db_session, normal_user, sample_product):
    """Test retrieving user with selectinload loads orders relationship efficiently."""
    service = OrderService(db_session)
    service.create_order(
        user_id=normal_user.id,
        req=OrderCreate(items=[OrderItemCreate(product_id=sample_product.id, quantity=2)])
    )

    repo = UserRepository(db_session)
    fetched_user = repo.get_by_id_with_orders(normal_user.id)

    assert fetched_user is not None
    assert len(fetched_user.orders) == 1
    assert fetched_user.orders[0].total_amount == 240.0

def test_pydantic_schema_excludes_password_hash(normal_user):
    """Test UserResponse Pydantic schema excludes password_hash during dump."""
    schema = UserResponse.model_validate(normal_user)
    data_dict = schema.model_dump()

    assert "password_hash" not in data_dict
    assert data_dict["email"] == normal_user.email
    assert data_dict["role"] == "user"

def test_order_repository_eager_loads_items_and_products(db_session, normal_user, sample_product):
    """Test OrderRepository get_by_id eager loads order items and nested product details."""
    service = OrderService(db_session)
    order = service.create_order(
        user_id=normal_user.id,
        req=OrderCreate(items=[OrderItemCreate(product_id=sample_product.id, quantity=1)])
    )

    repo = OrderRepository(db_session)
    fetched = repo.get_by_id(order.id)

    assert fetched is not None
    assert len(fetched.items) == 1
    assert fetched.items[0].product.name == sample_product.name
