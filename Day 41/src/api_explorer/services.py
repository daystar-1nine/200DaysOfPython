# ==============================================================================
# Program    : API Domain Services (services.py)
# Objective  : UserService and PostService mapping raw HTTP JSON responses to typed models.
# Concept    : Layered Service Architecture
# Why Used   : Converts raw dictionary data into strongly typed User & Post domain models.
# ==============================================================================

import os
import sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from api_explorer.client import APIClient
from api_explorer.models import User, Post

class UserService:
    """Service handling User business operations."""
    def __init__(self, client: APIClient):
        self.client = client

    def list_users(self) -> list[User]:
        raw_users = self.client.users.list()
        return [User.from_dict(u) for u in raw_users]

    def get_user(self, user_id: int) -> User:
        raw = self.client.users.get(user_id)
        return User.from_dict(raw)

    def search_users(self, query: str) -> list[User]:
        query_clean = query.strip().lower()
        all_users = self.list_users()
        return [
            u for u in all_users
            if query_clean in u.name.lower() or query_clean in u.username.lower() or query_clean in u.email.lower()
        ]

class PostService:
    """Service handling Post business operations."""
    def __init__(self, client: APIClient):
        self.client = client

    def list_posts(self, user_id: int | None = None) -> list[Post]:
        raw_posts = self.client.posts.list(user_id=user_id)
        return [Post.from_dict(p) for p in raw_posts]

    def get_post(self, post_id: int) -> Post:
        raw = self.client.posts.get(post_id)
        return Post.from_dict(raw)

    def create_post(self, title: str, body: str, user_id: int) -> Post:
        raw = self.client.posts.create(title=title, body=body, user_id=user_id)
        return Post.from_dict(raw)

    def update_post(self, post_id: int, title: str, body: str, user_id: int) -> Post:
        raw = self.client.posts.update(post_id=post_id, title=title, body=body, user_id=user_id)
        return Post.from_dict(raw)

    def patch_post(self, post_id: int, title: str | None = None, body: str | None = None) -> Post:
        raw = self.client.posts.patch(post_id=post_id, title=title, body=body)
        return Post.from_dict(raw)

    def delete_post(self, post_id: int) -> bool:
        self.client.posts.delete(post_id)
        return True
