# ==============================================================================
# Test Suite : Transactional Order Placement Tests (test_orders.py)
# Objective  : Integration testing of order placement, stock reduction, out-of-stock errors, and transaction rollback.
# Concept    : Database Transaction & Atomic Inventory Testing (Day 45 requirement)
# Why Used   : Asserts atomic commit/rollback guarantees during order checkout.
# ==============================================================================

def test_create_order_success(client):
    res_u = client.post("/users", json={"name": "Suraj", "email": "suraj@example.com"})
    user_id = res_u.json()["id"]

    res_p1 = client.post("/products", json={"name": "Keyboard", "price": 100.0, "stock": 10})
    p1_id = res_p1.json()["id"]

    res_p2 = client.post("/products", json={"name": "Mouse", "price": 50.0, "stock": 5})
    p2_id = res_p2.json()["id"]

    order_payload = {
        "user_id": user_id,
        "items": [
            {"product_id": p1_id, "quantity": 2},
            {"product_id": p2_id, "quantity": 1}
        ]
    }

    res_o = client.post("/orders", json=order_payload)
    assert res_o.status_code == 201
    order_data = res_o.json()
    assert order_data["user_id"] == user_id
    assert order_data["total_amount"] == 250.0  # (2*100) + (1*50)
    assert len(order_data["items"]) == 2

    # Verify inventory stock was reduced
    assert client.get(f"/products/{p1_id}").json()["stock"] == 8
    assert client.get(f"/products/{p2_id}").json()["stock"] == 4

def test_create_order_invalid_user(client):
    res_p = client.post("/products", json={"name": "Keyboard", "price": 100.0, "stock": 10})
    p_id = res_p.json()["id"]

    res_o = client.post("/orders", json={"user_id": 999, "items": [{"product_id": p_id, "quantity": 1}]})
    assert res_o.status_code == 404

def test_create_order_invalid_product(client):
    res_u = client.post("/users", json={"name": "Suraj", "email": "suraj@example.com"})
    u_id = res_u.json()["id"]

    res_o = client.post("/orders", json={"user_id": u_id, "items": [{"product_id": 999, "quantity": 1}]})
    assert res_o.status_code == 404

def test_create_order_insufficient_stock(client):
    res_u = client.post("/users", json={"name": "Suraj", "email": "suraj@example.com"})
    u_id = res_u.json()["id"]

    res_p = client.post("/products", json={"name": "Low Stock Keyboard", "price": 100.0, "stock": 2})
    p_id = res_p.json()["id"]

    res_o = client.post("/orders", json={"user_id": u_id, "items": [{"product_id": p_id, "quantity": 5}]})
    assert res_o.status_code == 400
    assert "Insufficient stock" in res_o.json()["detail"]

    # Verify stock remained untouched
    assert client.get(f"/products/{p_id}").json()["stock"] == 2

def test_create_order_transaction_rollback_on_stock_error(client):
    res_u = client.post("/users", json={"name": "Suraj", "email": "suraj@example.com"})
    u_id = res_u.json()["id"]

    res_p1 = client.post("/products", json={"name": "Available Product", "price": 50.0, "stock": 10})
    p1_id = res_p1.json()["id"]

    res_p2 = client.post("/products", json={"name": "Out of Stock Product", "price": 100.0, "stock": 1})
    p2_id = res_p2.json()["id"]

    # Order item 1 is valid (qty 2 out of 10), item 2 requests qty 5 out of 1
    order_payload = {
        "user_id": u_id,
        "items": [
            {"product_id": p1_id, "quantity": 2},
            {"product_id": p2_id, "quantity": 5}
        ]
    }

    res_o = client.post("/orders", json=order_payload)
    assert res_o.status_code == 400

    # What is tested: Transaction Rollback Guarantee
    # Why it is tested: Verifies product 1 stock was NOT decremented despite item 1 coming before failing item 2
    assert client.get(f"/products/{p1_id}").json()["stock"] == 10
    assert client.get(f"/products/{p2_id}").json()["stock"] == 1

def test_get_user_orders(client):
    res_u = client.post("/users", json={"name": "Suraj", "email": "suraj@example.com"})
    u_id = res_u.json()["id"]

    res_p = client.post("/products", json={"name": "Keyboard", "price": 100.0, "stock": 10})
    p_id = res_p.json()["id"]

    client.post("/orders", json={"user_id": u_id, "items": [{"product_id": p_id, "quantity": 1}]})
    client.post("/orders", json={"user_id": u_id, "items": [{"product_id": p_id, "quantity": 2}]})

    res_uo = client.get(f"/users/{u_id}/orders")
    assert res_uo.status_code == 200
    orders = res_uo.json()
    assert len(orders) == 2
