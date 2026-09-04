# ==============================================================================
# Program    : Order Placement & Atomic Transaction Rollback Tests (test_orders.py)
# Objective  : Test order creation, inventory stock deduction, ownership isolation, and rollback safety.
# Concept    : Atomic Transactions & Resource Ownership Testing
# Why Used   : Proves that orders process atomically or fail completely without partial stock mutation.
# ==============================================================================

import os
import sys
from app.models.product import Product

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_create_order_success(client, sample_product, normal_user_headers):
    """Test placing an order deducts inventory stock and returns HTTP 201 Created."""
    initial_stock = sample_product.stock
    payload = {
        "items": [
            {"product_id": sample_product.id, "quantity": 2}
        ]
    }
    response = client.post("/orders", json=payload, headers=normal_user_headers)
    assert response.status_code == 201
    data = response.json()
    assert data["total_amount"] == 240.0
    assert len(data["items"]) == 1

    # Verify inventory stock was reduced from 10 to 8
    product_res = client.get(f"/products/{sample_product.id}")
    assert product_res.json()["stock"] == initial_stock - 2

def test_create_order_insufficient_stock_returns_400(client, sample_product, normal_user_headers):
    """Test ordering more units than available stock returns HTTP 400 Bad Request."""
    payload = {
        "items": [
            {"product_id": sample_product.id, "quantity": 9999}
        ]
    }
    response = client.post("/orders", json=payload, headers=normal_user_headers)
    assert response.status_code == 400
    assert "insufficient stock" in response.json()["detail"].lower()

def test_create_order_nonexistent_product_returns_404(client, normal_user_headers):
    """Test ordering non-existent product ID returns HTTP 404 Not Found."""
    payload = {
        "items": [
            {"product_id": 99999, "quantity": 1}
        ]
    }
    response = client.post("/orders", json=payload, headers=normal_user_headers)
    assert response.status_code == 404

def test_transaction_rollback_when_item_out_of_stock(client, db_session, normal_user_headers):
    """Test atomic transaction rollback when multi-item order encounters an out-of-stock product.

    Verifies:
    1. Product A stock is 10.
    2. Product B stock is 0.
    3. Order requested: A x 2, B x 1.
    4. Order fails with HTTP 400.
    5. Product A stock remains UNCHANGED at 10 (Zero partial stock mutations).
    """
    prod_a = Product(name="Item A Available", price=50.0, stock=10)
    prod_b = Product(name="Item B Out Of Stock", price=30.0, stock=0)
    db_session.add(prod_a)
    db_session.add(prod_b)
    db_session.commit()
    db_session.refresh(prod_a)
    db_session.refresh(prod_b)

    payload = {
        "items": [
            {"product_id": prod_a.id, "quantity": 2},
            {"product_id": prod_b.id, "quantity": 1}
        ]
    }
    response = client.post("/orders", json=payload, headers=normal_user_headers)
    assert response.status_code == 400

    # Verify Product A stock was NOT partially deducted (remains 10)
    fetched_a = db_session.get(Product, prod_a.id)
    assert fetched_a is not None
    assert fetched_a.stock == 10

def test_user_read_own_order_success(client, sample_product, normal_user_headers):
    """Test authenticated user reading their own order returns 200 OK."""
    order_payload = {"items": [{"product_id": sample_product.id, "quantity": 1}]}
    create_res = client.post("/orders", json=order_payload, headers=normal_user_headers)
    order_id = create_res.json()["id"]

    get_res = client.get(f"/orders/{order_id}", headers=normal_user_headers)
    assert get_res.status_code == 200
    assert get_res.json()["id"] == order_id

def test_user_read_another_users_order_returns_403(client, sample_product, normal_user_headers, admin_user, db_session):
    """Test user attempting to view another user's order receives HTTP 403 Forbidden."""
    # Create order belonging to user B (admin_user)
    from app.services.order_service import OrderService
    from app.schemas.order import OrderCreate, OrderItemCreate

    service = OrderService(db_session)
    order_b = service.create_order(
        user_id=admin_user.id,
        req=OrderCreate(items=[OrderItemCreate(product_id=sample_product.id, quantity=1)])
    )

    # Attempt to read order_b using normal_user headers
    response = client.get(f"/orders/{order_b.id}", headers=normal_user_headers)
    assert response.status_code == 403
    assert "permission" in response.json()["detail"].lower()

def test_admin_read_any_user_order_success(client, sample_product, normal_user, admin_user_headers, db_session):
    """Test administrator can read any user's order."""
    from app.services.order_service import OrderService
    from app.schemas.order import OrderCreate, OrderItemCreate

    service = OrderService(db_session)
    user_order = service.create_order(
        user_id=normal_user.id,
        req=OrderCreate(items=[OrderItemCreate(product_id=sample_product.id, quantity=1)])
    )

    response = client.get(f"/orders/{user_order.id}", headers=admin_user_headers)
    assert response.status_code == 200
    assert response.json()["id"] == user_order.id
