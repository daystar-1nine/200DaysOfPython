# ==============================================================================
# Test Suite : User API Endpoints Tests (test_users.py)
# Objective  : Integration testing of User CRUD endpoints.
# Concept    : API Integration Testing
# Why Used   : Asserts user registration and retrieval behavior.
# ==============================================================================

def test_api_read_root(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Mini E-Commerce Backend" in res.json()["message"]

def test_create_user(client):
    res = client.post("/users", json={"name": "Suraj Sawant", "email": "suraj@example.com"})
    assert res.status_code == 201
    assert res.json()["id"] == 1
    assert res.json()["email"] == "suraj@example.com"

def test_create_duplicate_user_email(client):
    client.post("/users", json={"name": "Suraj", "email": "suraj@example.com"})
    res = client.post("/users", json={"name": "Suraj Dup", "email": "suraj@example.com"})
    assert res.status_code == 409

def test_get_user(client):
    res_c = client.post("/users", json={"name": "Alex Mercer", "email": "alex@example.com"})
    user_id = res_c.json()["id"]
    res = client.get(f"/users/{user_id}")
    assert res.status_code == 200
    assert res.json()["name"] == "Alex Mercer"

def test_get_user_not_found(client):
    res = client.get("/users/999")
    assert res.status_code == 404

def test_list_users(client):
    client.post("/users", json={"name": "User 1", "email": "u1@example.com"})
    client.post("/users", json={"name": "User 2", "email": "u2@example.com"})
    res = client.get("/users")
    assert res.status_code == 200
    assert len(res.json()) == 2
