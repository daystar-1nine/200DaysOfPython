# ==============================================================================
# Program    : Task 9 - Reusable APIClient Class (task9_api_client_class.py)
# Objective  : Encapsulate HTTP GET, POST, PUT, PATCH, DELETE in reusable OOP client.
# Concept    : API Client Wrapper Class (Day 31 OOP + Day 37 Exceptions)
# Why Used   : Provides clean SDK methods for calling remote API endpoints.
# ==============================================================================

import requests

class APIClient:
    """Reusable HTTP API Client wrapper."""
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def get(self, endpoint: str, params: dict | None = None) -> list | dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def post(self, endpoint: str, data: dict) -> dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = requests.post(url, json=data, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def put(self, endpoint: str, data: dict) -> dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = requests.put(url, json=data, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def patch(self, endpoint: str, data: dict) -> dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = requests.patch(url, json=data, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def delete(self, endpoint: str) -> dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = requests.delete(url, timeout=10)
        resp.raise_for_status()
        return resp.json()

if __name__ == "__main__":
    client = APIClient("https://jsonplaceholder.typicode.com")
    users = client.get("/users")
    print(f"APIClient fetched {len(users)} users. First user: {users[0]['name']}")
