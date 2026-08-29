# ==============================================================================
# Test Suite : Product API Endpoints Tests (test_products.py)
# Objective  : Integration testing of Product CRUD endpoints.
# Concept    : Product Inventory Testing
# Why Used   : Asserts product creation, update, and deletion behavior.
# ==============================================================================

def test_create_product(client):
    res = client.post("/products", json={"name": "Mechanical Keyboard", "price": 79.99, "stock": 10})
    assert res.status_code == 201
    assert res.json()["name"] == "Mechanical Keyboard"
    assert res.json()["stock"] == 10

def test_get_product(client):
    res_c = client.post("/products", json={"name": "Gaming Mouse", "price": 29.99, "stock": 15})
    prod_id = res_c.json()["id"]
    res = client.get(f"/products/{prod_id}")
    assert res.status_code == 200
    assert res.json()["price"] == 29.99

def test_get_product_not_found(client):
    res = client.get("/products/999")
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
    res = client.put(f"/products/{prod_id}", json={"name": "P1 Updated", "price": 12.5, "stock": 10})
    assert res.status_code == 200
    assert res.json()["price"] == 12.5

def test_patch_product(client):
    res_c = client.post("/products", json={"name": "P1", "price": 10.0, "stock": 5})
    prod_id = res_c.json()["id"]
    res = client.patch(f"/products/{prod_id}", json={"stock": 20})
    assert res.status_code == 200
    assert res.json()["stock"] == 20

def test_delete_product(client):
    res_c = client.post("/products", json={"name": "P1", "price": 10.0, "stock": 5})
    prod_id = res_c.json()["id"]
    res = client.delete(f"/products/{prod_id}")
    assert res.status_code == 200
    assert client.get(f"/products/{prod_id}").status_code == 404
