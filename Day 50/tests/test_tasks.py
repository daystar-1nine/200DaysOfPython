"""
===============================================================================
DAY 50 — TASK CRUD, FILTERING & ISOLATION INTEGRATION TESTS
===============================================================================
This module tests task creation, reading, full update (PUT), partial patch (PATCH),
deletion, status/priority filtering, search, pagination, edge cases, and user isolation.
===============================================================================
"""

from typing import Dict
from fastapi.testclient import TestClient
from app.models.user import User


def test_create_task_success(client: TestClient, user_auth_headers: Dict[str, str]) -> None:
    """Test creating a new task item (POST /tasks)."""
    payload = {
        "title": "Build TaskFlow API",
        "description": "Complete Day 50 milestone project.",
        "status": "TODO",
        "priority": "HIGH",
    }
    response = client.post("/tasks", json=payload, headers=user_auth_headers)
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "Build TaskFlow API"
    assert data["status"] == "TODO"
    assert data["priority"] == "HIGH"
    assert "id" in data
    assert "created_at" in data


def test_create_task_empty_title_fails(client: TestClient, user_auth_headers: Dict[str, str]) -> None:
    """Test creating a task with empty title fails with 422."""
    payload = {"title": "", "status": "TODO"}
    response = client.post("/tasks", json=payload, headers=user_auth_headers)
    assert response.status_code == 422


def test_create_task_invalid_status_fails(client: TestClient, user_auth_headers: Dict[str, str]) -> None:
    """Test creating a task with invalid status enum fails with 422."""
    payload = {"title": "Task", "status": "INVALID_STATUS"}
    response = client.post("/tasks", json=payload, headers=user_auth_headers)
    assert response.status_code == 422


def test_create_task_invalid_priority_fails(client: TestClient, user_auth_headers: Dict[str, str]) -> None:
    """Test creating a task with invalid priority enum fails with 422."""
    payload = {"title": "Task", "priority": "SUPER_HIGH"}
    response = client.post("/tasks", json=payload, headers=user_auth_headers)
    assert response.status_code == 422


def test_list_tasks_user_isolation(
    client: TestClient,
    normal_user: User,
    user_auth_headers: Dict[str, str],
    other_user: User,
    other_user_auth_headers: Dict[str, str],
) -> None:
    """Test User A cannot view User B's tasks when listing /tasks."""
    client.post("/tasks", json={"title": "User A Task"}, headers=user_auth_headers)
    client.post("/tasks", json={"title": "User B Task"}, headers=other_user_auth_headers)

    res_a = client.get("/tasks", headers=user_auth_headers)
    assert res_a.status_code == 200
    tasks_a = res_a.json()
    assert len(tasks_a) == 1
    assert tasks_a[0]["title"] == "User A Task"

    res_b = client.get("/tasks", headers=other_user_auth_headers)
    assert res_b.status_code == 200
    tasks_b = res_b.json()
    assert len(tasks_b) == 1
    assert tasks_b[0]["title"] == "User B Task"


def test_filter_tasks_by_status(client: TestClient, user_auth_headers: Dict[str, str]) -> None:
    """Test filtering user tasks by status (GET /tasks?status=COMPLETED)."""
    client.post("/tasks", json={"title": "Task 1", "status": "TODO"}, headers=user_auth_headers)
    client.post("/tasks", json={"title": "Task 2", "status": "COMPLETED"}, headers=user_auth_headers)

    res = client.get("/tasks?status=COMPLETED", headers=user_auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert data[0]["title"] == "Task 2"
    assert data[0]["status"] == "COMPLETED"


def test_filter_tasks_by_priority(client: TestClient, user_auth_headers: Dict[str, str]) -> None:
    """Test filtering user tasks by priority (GET /tasks?priority=HIGH)."""
    client.post("/tasks", json={"title": "Low Priority", "priority": "LOW"}, headers=user_auth_headers)
    client.post("/tasks", json={"title": "High Priority", "priority": "HIGH"}, headers=user_auth_headers)

    res = client.get("/tasks?priority=HIGH", headers=user_auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert data[0]["title"] == "High Priority"


def test_search_tasks_by_query(client: TestClient, user_auth_headers: Dict[str, str]) -> None:
    """Test searching tasks by title or description query (GET /tasks?search=python)."""
    client.post("/tasks", json={"title": "Learn Python Backend", "description": "FastAPI and SQL"}, headers=user_auth_headers)
    client.post("/tasks", json={"title": "Buy Groceries", "description": "Milk and bread"}, headers=user_auth_headers)

    res = client.get("/tasks?search=python", headers=user_auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert "Python" in data[0]["title"]


def test_get_task_by_id_owner_success(client: TestClient, user_auth_headers: Dict[str, str]) -> None:
    """Test task owner fetching single task detail by ID."""
    res_create = client.post("/tasks", json={"title": "Single Task"}, headers=user_auth_headers)
    task_id = res_create.json()["id"]

    res_get = client.get(f"/tasks/{task_id}", headers=user_auth_headers)
    assert res_get.status_code == 200
    assert res_get.json()["id"] == task_id


def test_get_task_by_id_other_user_forbidden(
    client: TestClient,
    user_auth_headers: Dict[str, str],
    other_user_auth_headers: Dict[str, str],
) -> None:
    """Test User B trying to access User A's task by ID returns 403 Forbidden."""
    res_create = client.post("/tasks", json={"title": "User A Secret Task"}, headers=user_auth_headers)
    task_id = res_create.json()["id"]

    res_get = client.get(f"/tasks/{task_id}", headers=other_user_auth_headers)
    assert res_get.status_code == 403
    assert res_get.json()["error"]["code"] == "FORBIDDEN"


def test_put_update_task_success(client: TestClient, user_auth_headers: Dict[str, str]) -> None:
    """Test full task replacement (PUT /tasks/{id})."""
    res_create = client.post("/tasks", json={"title": "Old Title", "status": "TODO"}, headers=user_auth_headers)
    task_id = res_create.json()["id"]

    put_payload = {
        "title": "New Replaced Title",
        "description": "Updated description",
        "status": "IN_PROGRESS",
        "priority": "HIGH",
        "due_date": None,
    }
    res_put = client.put(f"/tasks/{task_id}", json=put_payload, headers=user_auth_headers)
    assert res_put.status_code == 200
    data = res_put.json()
    assert data["title"] == "New Replaced Title"
    assert data["status"] == "IN_PROGRESS"


def test_put_update_nonexistent_task_fails(client: TestClient, user_auth_headers: Dict[str, str]) -> None:
    """Test PUT update on non-existent task ID returns 404."""
    put_payload = {
        "title": "Title",
        "description": "Desc",
        "status": "TODO",
        "priority": "LOW",
        "due_date": None,
    }
    res_put = client.put("/tasks/99999", json=put_payload, headers=user_auth_headers)
    assert res_put.status_code == 404


def test_patch_update_task_success(client: TestClient, user_auth_headers: Dict[str, str]) -> None:
    """Test partial task update (PATCH /tasks/{id})."""
    res_create = client.post("/tasks", json={"title": "Task To Patch", "status": "TODO"}, headers=user_auth_headers)
    task_id = res_create.json()["id"]

    patch_payload = {"status": "COMPLETED"}
    res_patch = client.patch(f"/tasks/{task_id}", json=patch_payload, headers=user_auth_headers)
    assert res_patch.status_code == 200
    data = res_patch.json()
    assert data["title"] == "Task To Patch"
    assert data["status"] == "COMPLETED"


def test_patch_nonexistent_task_fails(client: TestClient, user_auth_headers: Dict[str, str]) -> None:
    """Test PATCH update on non-existent task ID returns 404."""
    res_patch = client.patch("/tasks/99999", json={"status": "COMPLETED"}, headers=user_auth_headers)
    assert res_patch.status_code == 404


def test_delete_task_owner_success(client: TestClient, user_auth_headers: Dict[str, str]) -> None:
    """Test task owner deleting task by ID (DELETE /tasks/{id})."""
    res_create = client.post("/tasks", json={"title": "Task To Delete"}, headers=user_auth_headers)
    task_id = res_create.json()["id"]

    res_del = client.delete(f"/tasks/{task_id}", headers=user_auth_headers)
    assert res_del.status_code == 204

    res_get = client.get(f"/tasks/{task_id}", headers=user_auth_headers)
    assert res_get.status_code == 404


def test_delete_task_other_user_forbidden(
    client: TestClient,
    user_auth_headers: Dict[str, str],
    other_user_auth_headers: Dict[str, str],
) -> None:
    """Test User B trying to delete User A's task returns 403 Forbidden."""
    res_create = client.post("/tasks", json={"title": "User A Protected Task"}, headers=user_auth_headers)
    task_id = res_create.json()["id"]

    res_del = client.delete(f"/tasks/{task_id}", headers=other_user_auth_headers)
    assert res_del.status_code == 403


def test_delete_nonexistent_task_fails(client: TestClient, user_auth_headers: Dict[str, str]) -> None:
    """Test deleting non-existent task ID returns 404."""
    res_del = client.delete("/tasks/99999", headers=user_auth_headers)
    assert res_del.status_code == 404


def test_task_pagination(client: TestClient, user_auth_headers: Dict[str, str]) -> None:
    """Test task list pagination offset and limit parameters."""
    for i in range(5):
        client.post("/tasks", json={"title": f"Item {i}"}, headers=user_auth_headers)

    res = client.get("/tasks?offset=0&limit=2", headers=user_auth_headers)
    assert res.status_code == 200
    assert len(res.json()) == 2
