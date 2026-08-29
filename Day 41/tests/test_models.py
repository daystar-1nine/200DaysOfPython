# ==============================================================================
# Test Suite : API Response Domain Models Unit Tests (test_models.py)
# Objective  : Test User and Post dataclasses, factory methods, and dunder protocols.
# Concept    : Unit Testing Dataclasses & Dunder Protocols (__str__, __repr__, __getitem__)
# Why Used   : Asserts domain model creation and formatting behaviors.
# ==============================================================================

import os
import sys
import unittest

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from api_explorer.models import User, Post

def test_user_dataclass_fields():
    u = User(id=1, name="Suraj", username="daystar", email="suraj@example.com", city="Mumbai", company="Google")
    assert u.id == 1
    assert u.name == "Suraj"
    assert u.city == "Mumbai"

def test_user_from_dict_factory():
    raw = {
        "id": 1,
        "name": "Leanne Graham",
        "username": "Bret",
        "email": "Sincere@april.biz",
        "address": {"city": "Gwenborough"},
        "company": {"name": "Romaguera-Crona"}
    }
    u = User.from_dict(raw)
    assert u.id == 1
    assert u.city == "Gwenborough"
    assert u.company == "Romaguera-Crona"

def test_user_dunder_str():
    u = User(id=1, name="Suraj", username="daystar", email="suraj@example.com", city="Mumbai")
    s = str(u)
    assert "USER DETAILS #1" in s
    assert "Name     : Suraj" in s

def test_user_dunder_repr():
    u = User(id=1, name="Suraj", username="daystar", email="suraj@example.com")
    r = repr(u)
    assert "<User id=1 name='Suraj'" in r

def test_user_dunder_getitem():
    u = User(id=1, name="Suraj", username="daystar", email="suraj@example.com")
    assert u["name"] == "Suraj"
    assert u["email"] == "suraj@example.com"

def test_post_dataclass_fields():
    p = Post(id=1, title="Test Title", body="Test Body", user_id=10)
    assert p.id == 1
    assert p.title == "Test Title"
    assert p.user_id == 10

def test_post_from_dict_factory():
    raw = {"id": 2, "title": "Sample", "body": "Body text", "userId": 5}
    p = Post.from_dict(raw)
    assert p.id == 2
    assert p.user_id == 5

def test_post_dunder_str():
    p = Post(id=1, title="Test Title", body="Test Body", user_id=10)
    s = str(p)
    assert "POST #1 (User #10)" in s
    assert "Title : Test Title" in s

def test_post_dunder_getitem():
    p = Post(id=1, title="Test Title", body="Test Body", user_id=10)
    assert p["title"] == "Test Title"

class TestModelsRunner(unittest.TestCase):
    def test_models_standalone(self):
        u = User(id=1, name="A", username="B", email="C")
        self.assertEqual(u.id, 1)

if __name__ == "__main__":
    unittest.main()
