# ==============================================================================
# Test Suite : Domain Services Unit Tests (test_services.py)
# Objective  : Test UserService and PostService models mapping and business logic.
# Concept    : Layered Architecture Unit Testing with Mocks
# Why Used   : Asserts business logic without real HTTP calls.
# ==============================================================================

import os
import sys
import unittest
from unittest.mock import MagicMock

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from api_explorer.client import APIClient, UsersResource, PostsResource
from api_explorer.services import UserService, PostService

def test_user_service_list_users():
    mock_client = MagicMock(spec=APIClient)
    mock_users_res = MagicMock(spec=UsersResource)
    mock_client.users = mock_users_res

    mock_users_res.list.return_value = [
        {"id": 1, "name": "Leanne Graham", "username": "Bret", "email": "leanne@example.com"},
        {"id": 2, "name": "Ervin Howell", "username": "Antonette", "email": "ervin@example.com"}
    ]

    service = UserService(mock_client)
    users = service.list_users()

    assert len(users) == 2
    assert users[0].id == 1
    assert users[0].name == "Leanne Graham"

def test_user_service_search_users():
    mock_client = MagicMock(spec=APIClient)
    mock_users_res = MagicMock(spec=UsersResource)
    mock_client.users = mock_users_res

    mock_users_res.list.return_value = [
        {"id": 1, "name": "Leanne Graham", "username": "Bret", "email": "leanne@example.com"},
        {"id": 2, "name": "Ervin Howell", "username": "Antonette", "email": "ervin@example.com"}
    ]

    service = UserService(mock_client)
    matches = service.search_users("ervin")

    assert len(matches) == 1
    assert matches[0].id == 2

def test_post_service_create_post():
    mock_client = MagicMock(spec=APIClient)
    mock_posts_res = MagicMock(spec=PostsResource)
    mock_client.posts = mock_posts_res

    mock_posts_res.create.return_value = {
        "id": 101,
        "title": "Created Title",
        "body": "Created Body",
        "userId": 1
    }

    service = PostService(mock_client)
    post = service.create_post("Created Title", "Created Body", user_id=1)

    assert post.id == 101
    assert post.title == "Created Title"
    assert post.user_id == 1

def test_post_service_delete_post():
    mock_client = MagicMock(spec=APIClient)
    mock_posts_res = MagicMock(spec=PostsResource)
    mock_client.posts = mock_posts_res

    mock_posts_res.delete.return_value = {}

    service = PostService(mock_client)
    result = service.delete_post(1)

    assert result is True
    mock_posts_res.delete.assert_called_once_with(1)

class TestServicesRunner(unittest.TestCase):
    def test_services_standalone(self):
        mock_client = MagicMock(spec=APIClient)
        service = UserService(mock_client)
        self.assertIsNotNone(service)

if __name__ == "__main__":
    unittest.main()
