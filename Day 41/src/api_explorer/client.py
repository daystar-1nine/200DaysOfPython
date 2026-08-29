# ==============================================================================
# Program    : HTTP API Client & Resource SDK (client.py)
# Objective  : Resilient API Client wrapper and Bonus SDK resources (client.users, client.posts).
# Concept    : SDK Design (Day 31 OOP) + @timer profiling (Day 33) + API Exception Chaining (Day 37)
# Why Used   : Encapsulates network calls into clean SDK resources (client.users.list()).
# ==============================================================================

from functools import wraps
import os
import sys
import time
from typing import Any, Callable
import requests

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from api_explorer.exceptions import (
    APIError,
    APIConnectionError,
    APINotFoundError,
    APITimeoutError,
    APIValidationError
)

# What is used : @timer Decorator (Day 33 requirement)
# Why it is used: Profiles execution duration of HTTP network requests
def timer(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_t = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_t
        wrapper.last_execution_time = elapsed  # type: ignore
        return result
    wrapper.last_execution_time = 0.0  # type: ignore
    return wrapper

class APIClient:
    """Core HTTP Client wrapper for executing REST operations."""
    def __init__(self, base_url: str = "https://jsonplaceholder.typicode.com", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "REST-API-Explorer/1.0",
            "Accept": "application/json"
        })
        # Bonus SDK Resources
        self.users = UsersResource(self)
        self.posts = PostsResource(self)

    @timer
    def request(self, method: str, endpoint: str, params: dict[str, Any] | None = None, json_data: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = self.session.request(
                method=method.upper(),
                url=url,
                params=params,
                json=json_data,
                timeout=self.timeout
            )
            if response.status_code == 404:
                raise APINotFoundError(f"Resource at '{url}' was not found.")
            if response.status_code == 422:
                raise APIValidationError(f"Validation error for payload at '{url}'.")

            response.raise_for_status()
            if response.status_code == 204 or not response.text.strip():
                return {}
            return response.json()

        except requests.exceptions.Timeout as e:
            raise APITimeoutError(f"Request to '{url}' timed out after {self.timeout}s.") from e
        except requests.exceptions.ConnectionError as e:
            raise APIConnectionError(f"Failed connecting to server at '{url}'.") from e
        except requests.exceptions.HTTPError as e:
            raise APIError(f"HTTP Error {response.status_code} for '{url}': {e}", status_code=response.status_code) from e
        except requests.exceptions.RequestException as e:
            raise APIError(f"Network request failed for '{url}': {e}") from e

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        return self.request("GET", endpoint, params=params)

    def post(self, endpoint: str, data: dict[str, Any]) -> Any:
        return self.request("POST", endpoint, json_data=data)

    def put(self, endpoint: str, data: dict[str, Any]) -> Any:
        return self.request("PUT", endpoint, json_data=data)

    def patch(self, endpoint: str, data: dict[str, Any]) -> Any:
        return self.request("PATCH", endpoint, json_data=data)

    def delete(self, endpoint: str) -> Any:
        return self.request("DELETE", endpoint)


class UsersResource:
    """Bonus SDK Resource for User API endpoints."""
    def __init__(self, client: APIClient):
        self._client = client

    def list(self) -> list[dict[str, Any]]:
        return self._client.get("/users")

    def get(self, user_id: int) -> dict[str, Any]:
        return self._client.get(f"/users/{user_id}")


class PostsResource:
    """Bonus SDK Resource for Post API endpoints."""
    def __init__(self, client: APIClient):
        self._client = client

    def list(self, user_id: int | None = None) -> list[dict[str, Any]]:
        params = {"userId": user_id} if user_id else None
        return self._client.get("/posts", params=params)

    def get(self, post_id: int) -> dict[str, Any]:
        return self._client.get(f"/posts/{post_id}")

    def create(self, title: str, body: str, user_id: int) -> dict[str, Any]:
        payload = {"title": title, "body": body, "userId": user_id}
        return self._client.post("/posts", data=payload)

    def update(self, post_id: int, title: str, body: str, user_id: int) -> dict[str, Any]:
        payload = {"id": post_id, "title": title, "body": body, "userId": user_id}
        return self._client.put(f"/posts/{post_id}", data=payload)

    def patch(self, post_id: int, title: str | None = None, body: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if title is not None:
            payload["title"] = title
        if body is not None:
            payload["body"] = body
        return self._client.patch(f"/posts/{post_id}", data=payload)

    def delete(self, post_id: int) -> dict[str, Any]:
        return self._client.delete(f"/posts/{post_id}")
