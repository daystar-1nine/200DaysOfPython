# ==============================================================================
# Test Suite : Pydantic Models Validation Tests (test_validation.py)
# Objective  : Test Pydantic UserCreate, UserUpdate, UserPatch schema validation rules.
# Concept    : Unit Testing Pydantic Field Constraints
# Why Used   : Asserts request schema validation raises ValidationError on bad data.
# ==============================================================================

import os
import sys
import pytest
from pydantic import ValidationError

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.models.user import UserCreate, UserPatch

def test_user_create_valid():
    u = UserCreate(name="Suraj", email="suraj@example.com", age=21)
    assert u.name == "Suraj"
    assert u.age == 21

def test_user_create_short_name_raises_error():
    with pytest.raises(ValidationError):
        UserCreate(name="A", email="a@example.com", age=20)

def test_user_create_negative_age_raises_error():
    with pytest.raises(ValidationError):
        UserCreate(name="Suraj", email="suraj@example.com", age=-5)

def test_user_create_excessive_age_raises_error():
    with pytest.raises(ValidationError):
        UserCreate(name="Suraj", email="suraj@example.com", age=150)

def test_user_patch_optional_fields():
    p = UserPatch(age=25)
    assert p.name is None
    assert p.email is None
    assert p.age == 25
