# ==============================================================================
# Test Suite : Order Placement & Ownership Isolation Tests (test_orders.py)
# Objective  : Test order checkout, inventory deductions, and resource ownership enforcement.
# Concept    : Resource Ownership & Transaction Testing
# Why Used   : Ensures users can only access their own orders and cross-user URL tampering fails with 403.
# ==============================================================================

import os
import sys
import pytest

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.security import create_access_token

def test_create_order_success(client, user_token_headers, admin_token_headers):
    # Admin creates product
    p_res = client.post("/products", headers=admin_token_headers, json={
        "name": "Wireless Mouse",
        "price": 25.00,
        "stock": 10
    })
    product_id = p_res.json()["id"]

    # User places order
    order_res = client.post("/orders", headers=user_token_headers, json={
        "items": [{"product_id": product_id, "quantity": 2}]
    })
    assert order_res.status_code == 201
    data = order_res.json()
    assert data["total_amount"] == 50.00
    assert len(data["items"]) == 1

    # Verify product stock deducted from 10 to 8
    p_check = client.get(f"/products/{product_id}")
    assert p_check.json()["stock"] == 8

def test_create_order_insufficient_stock_fails(client, user_token_headers, admin_token_headers):
    p_res = client.post("/products", headers=admin_token_headers, json={
        "name": "Rare Monitor",
        "price": 300.00,
        "stock": 1
    })
    product_id = p_res.json()["id"]

    order_res = client.post("/orders", headers=user_token_headers, json={
        "items": [{"product_id": product_id, "quantity": 5}]
    })
    assert order_res.status_code == 400
    assert "Insufficient stock" in order_res.json()["detail"]

def test_user_can_access_own_order(client, test_user, user_token_headers, admin_token_headers):
    p_res = client.post("/products", headers=admin_token_headers, json={
        "name": "Keyboard",
        "price": 80.00,
        "stock": 5
    })
    product_id = p_res.json()["id"]

    order_res = client.post("/orders", headers=user_token_headers, json={
        "items": [{"product_id": product_id, "quantity": 1}]
    })
    order_id = order_res.json()["id"]

    get_res = client.get(f"/orders/{order_id}", headers=user_token_headers)
    assert get_res.status_code == 200
    assert get_res.json()["id"] == order_id

def test_user_cannot_access_another_users_order(client, test_user, user_token_headers, admin_token_headers):
    # Order placed by test_user
    p_res = client.post("/products", headers=admin_token_headers, json={
        "name": "Desk Mat",
        "price": 20.00,
        "stock": 5
    })
    product_id = p_res.json()["id"]

    order_res = client.post("/orders", headers=user_token_headers, json={
        "items": [{"product_id": product_id, "quantity": 1}]
    })
    order_id = order_res.json()["id"]

    # Register second user
    client.post("/auth/register", json={
        "name": "Attacker User",
        "email": "attacker@example.com",
        "password": "Password123!"
    })
    login_res = client.post("/auth/login", json={
        "email": "attacker@example.com",
        "password": "Password123!"
    })
    attacker_token = login_res.json()["access_token"]
    attacker_headers = {"Authorization": f"Bearer {attacker_token}"}

    # Attacker attempts to view test_user's order -> 403 Forbidden
    attempt_res = client.get(f"/orders/{order_id}", headers=attacker_headers)
    assert attempt_res.status_code == 403
    assert "not authorized to view this order" in attempt_res.json()["detail"]

def test_admin_can_access_any_order(client, user_token_headers, admin_token_headers):
    p_res = client.post("/products", headers=admin_token_headers, json={
        "name": "Webcam",
        "price": 60.00,
        "stock": 5
    })
    product_id = p_res.json()["id"]

    order_res = client.post("/orders", headers=user_token_headers, json={
        "items": [{"product_id": product_id, "quantity": 1}]
    })
    order_id = order_res.json()["id"]

    admin_get_res = client.get(f"/orders/{order_id}", headers=admin_token_headers)
    assert admin_get_res.status_code == 200
    assert admin_get_res.json()["id"] == order_id
