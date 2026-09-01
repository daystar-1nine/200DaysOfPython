# ==============================================================================
# Test Suite : Product API Endpoints Tests (test_products.py)
# Objective  : Integration testing of Product CRUD endpoints with category & description fields.
# Concept    : Product Inventory & Catalog Testing
# Why Used   : Asserts product creation, replacement, patch, and deletion.
# ==============================================================================

def test_create_product(client):
    payload = {
        "name": "Mechanical Keyboard",
        "price": 79.99,
        "stock": 10,
        "description": "RGB Backlit Mechanical Keyboard",
        "category": "Electronics"
    }
    res = client.post("/products", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["name"] == "Mechanical Keyboard"
    assert data["category"] == "Electronics"
    assert data["description"] == "RGB Backlit Mechanical Keyboard"

def test_get_product(client):
    res_c = client.post("/products", json={"name": "Gaming Mouse", "price": 29.99, "stock": 15, "category": "Electronics"})
    prod_id = res_c.json()["id"]
    res = client.get(f"/products/{prod_id}")
    assert res.status_code == 200
    assert res.json()["price"] == 29.99

def test_get_product_not_found(client):
    res = client.get("/products/9999")
    assert res.status_code == 404

def test_list_products(client):
    client.post("/products", json={"name": "P1", "price": 10.0, "stock": 5})
    client.post("/products", json={"name": "P2", "price": 20.0, "stock": 8})
    res = client.get("/products")
    assert res.status_code == 200
    assert len(res.json()) == 2

def test_replace_product(client):
    res_c = client.post("/products", json={"name": "P1", "price": 10.0, "stock": 5})
    prod_id = res_c.json()["id"]
    res = client.put(f"/products/{prod_id}", json={"name": "P1 Updated", "price": 12.5, "stock": 10, "category": "Hardware"})
    assert res.status_code == 200
    assert res.json()["category"] == "Hardware"

def test_patch_product(client):
    res_c = client.post("/products", json={"name": "P1", "price": 10.0, "stock": 5})
    prod_id = res_c.json()["id"]
    res = client.patch(f"/products/{prod_id}", json={"category": "Peripherals"})
    assert res.status_code == 200
    assert res.json()["category"] == "Peripherals"

def test_delete_product(client):
    res_c = client.post("/products", json={"name": "P1", "price": 10.0, "stock": 5})
    prod_id = res_c.json()["id"]
    res = client.delete(f"/products/{prod_id}")
    assert res.status_code == 200
    assert client.get(f"/products/{prod_id}").status_code == 404
