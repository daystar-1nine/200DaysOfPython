# ==============================================================================
# Program    : Product Catalog Integration & RBAC Tests (test_products.py)
# Objective  : Test public product reading and admin-only product mutations (create, update, delete).
# Concept    : Endpoint Permission Boundaries
# Why Used   : Verifies public read availability while restricting catalog mutations to administrators.
# ==============================================================================

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_list_products_public_success(client, sample_product):
    """Test public listing of catalog products without authentication headers."""
    response = client.get("/products")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["name"] == sample_product.name

def test_get_product_by_id_public_success(client, sample_product):
    """Test public retrieval of a single product by ID."""
    response = client.get(f"/products/{sample_product.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == sample_product.name
    assert data["price"] == sample_product.price

def test_get_nonexistent_product_returns_404(client):
    """Test requesting non-existent product ID returns HTTP 404 Not Found with standardized error."""
    response = client.get("/products/999999")
    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "PRODUCT_NOT_FOUND"

def test_create_product_as_admin_success(client, admin_user_headers):
    """Test creating product as admin user returns HTTP 201 Created."""
    payload = {
        "name": "Wireless Gaming Mouse",
        "price": 79.99,
        "stock": 15,
        "description": "Ergonomic Gaming Mouse",
        "category": "Electronics"
    }
    response = client.post("/products", json=payload, headers=admin_user_headers)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Wireless Gaming Mouse"
    assert data["price"] == 79.99
    assert data["stock"] == 15

def test_create_product_as_normal_user_returns_403(client, normal_user_headers):
    """Test normal user attempting to create product returns HTTP 403 Forbidden."""
    payload = {
        "name": "Unauthorized Product",
        "price": 10.0,
        "stock": 5
    }
    response = client.post("/products", json=payload, headers=normal_user_headers)
    assert response.status_code == 403
    data = response.json()
    assert data["error"]["code"] == "FORBIDDEN"

def test_create_product_unauthenticated_returns_401(client):
    """Test creating product without authentication returns HTTP 401 Unauthorized."""
    payload = {
        "name": "Unauthenticated Product",
        "price": 10.0,
        "stock": 5
    }
    response = client.post("/products", json=payload)
    assert response.status_code == 401

def test_update_product_as_admin_success(client, sample_product, admin_user_headers):
    """Test admin updating product price and stock returns HTTP 200 OK."""
    payload = {"price": 99.99, "stock": 20}
    response = client.patch(f"/products/{sample_product.id}", json=payload, headers=admin_user_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["price"] == 99.99
    assert data["stock"] == 20

def test_update_product_as_normal_user_returns_403(client, sample_product, normal_user_headers):
    """Test normal user attempting to update product returns HTTP 403 Forbidden."""
    payload = {"price": 1.00}
    response = client.patch(f"/products/{sample_product.id}", json=payload, headers=normal_user_headers)
    assert response.status_code == 403

def test_delete_product_as_admin_success(client, sample_product, admin_user_headers):
    """Test admin deleting product returns HTTP 204 No Content."""
    response = client.delete(f"/products/{sample_product.id}", headers=admin_user_headers)
    assert response.status_code == 204

    # Verify product was deleted
    get_res = client.get(f"/products/{sample_product.id}")
    assert get_res.status_code == 404

def test_delete_product_as_normal_user_returns_403(client, sample_product, normal_user_headers):
    """Test normal user attempting to delete product returns HTTP 403 Forbidden."""
    response = client.delete(f"/products/{sample_product.id}", headers=normal_user_headers)
    assert response.status_code == 403
