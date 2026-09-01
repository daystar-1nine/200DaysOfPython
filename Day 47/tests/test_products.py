# ==============================================================================
# Test Suite : Product Catalog Authorization Tests (test_products.py)
# Objective  : Test public catalog reading vs admin-only creation/updating/deletion.
# Concept    : Role-Based Access Control on Resource Mutation
# Why Used   : Asserts non-admins receive 403 Forbidden when attempting to mutate catalog.
# ==============================================================================

import os
import sys
import pytest

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

def test_list_products_public_access(client):
    response = client.get("/products")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_create_product_admin_success(client, admin_token_headers):
    response = client.post("/products", headers=admin_token_headers, json={
        "name": "Gaming Laptop",
        "price": 1499.99,
        "stock": 10,
        "description": "High performance laptop",
        "category": "Electronics"
    })
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Gaming Laptop"
    assert data["price"] == 1499.99

def test_create_product_regular_user_forbidden(client, user_token_headers):
    response = client.post("/products", headers=user_token_headers, json={
        "name": "Unauthorized Laptop",
        "price": 999.00,
        "stock": 5
    })
    assert response.status_code == 403

def test_create_product_unauthenticated_fails(client):
    response = client.post("/products", json={
        "name": "Anonymous Laptop",
        "price": 999.00,
        "stock": 5
    })
    assert response.status_code == 401

def test_patch_product_admin_success(client, admin_token_headers):
    create_res = client.post("/products", headers=admin_token_headers, json={
        "name": "Headphones",
        "price": 199.99,
        "stock": 15
    })
    product_id = create_res.json()["id"]

    patch_res = client.patch(f"/products/{product_id}", headers=admin_token_headers, json={
        "price": 149.99,
        "stock": 20
    })
    assert patch_res.status_code == 200
    data = patch_res.json()
    assert data["price"] == 149.99
    assert data["stock"] == 20

def test_patch_product_regular_user_forbidden(client, user_token_headers, admin_token_headers):
    create_res = client.post("/products", headers=admin_token_headers, json={
        "name": "Smart Phone",
        "price": 799.99,
        "stock": 8
    })
    product_id = create_res.json()["id"]

    patch_res = client.patch(f"/products/{product_id}", headers=user_token_headers, json={"price": 1.00})
    assert patch_res.status_code == 403

def test_delete_product_admin_success(client, admin_token_headers):
    create_res = client.post("/products", headers=admin_token_headers, json={
        "name": "Item to delete",
        "price": 10.00,
        "stock": 1
    })
    product_id = create_res.json()["id"]

    delete_res = client.delete(f"/products/{product_id}", headers=admin_token_headers)
    assert delete_res.status_code == 204

def test_delete_product_regular_user_forbidden(client, user_token_headers, admin_token_headers):
    create_res = client.post("/products", headers=admin_token_headers, json={
        "name": "Item safe",
        "price": 10.00,
        "stock": 1
    })
    product_id = create_res.json()["id"]

    delete_res = client.delete(f"/products/{product_id}", headers=user_token_headers)
    assert delete_res.status_code == 403
