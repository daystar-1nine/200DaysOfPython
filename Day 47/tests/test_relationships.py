# ==============================================================================
# Test Suite : ORM Relationships & Schema Serialization Tests (test_relationships.py)
# Objective  : Test ORM relationships, eager loading, and Pydantic field sanitization.
# Concept    : Relational Integrity & Schema Validation
# Why Used   : Ensures user_id foreign keys, order cascade deletes, and schema responses work properly.
# ==============================================================================

import os
import sys
import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.database import Base
from app.models.user import User
from app.models.product import Product
from app.models.order import Order
from app.models.order_item import OrderItem
from app.schemas.user import UserResponse
from app.security import hash_password

class TestDatabaseRelationships(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)

    def test_user_orders_relationship_eager_loading(self):
        user = User(name="Suraj", email="suraj_rel@example.com", password_hash=hash_password("Pass123!"), role="user")
        self.db.add(user)
        self.db.commit()

        product = Product(name="Headphones", price=150.0, stock=5, category="Audio")
        self.db.add(product)
        self.db.commit()

        order = Order(user_id=user.id, total_amount=150.0, status="completed")
        self.db.add(order)
        self.db.commit()

        item = OrderItem(order_id=order.id, product_id=product.id, quantity=1, price=150.0)
        self.db.add(item)
        self.db.commit()

        fetched_user = self.db.query(User).filter_by(id=user.id).first()
        self.assertEqual(len(fetched_user.orders), 1)
        self.assertEqual(fetched_user.orders[0].total_amount, 150.0)

    def test_user_deletion_cascades_to_orders(self):
        user = User(name="ToDelete", email="delete_rel@example.com", password_hash=hash_password("Pass123!"))
        self.db.add(user)
        self.db.commit()

        order = Order(user_id=user.id, total_amount=50.0, status="pending")
        self.db.add(order)
        self.db.commit()

        self.assertEqual(self.db.query(Order).count(), 1)

        self.db.delete(user)
        self.db.commit()

        self.assertEqual(self.db.query(Order).count(), 0)

    def test_nested_pydantic_schema_serialization_excludes_hash(self):
        user = User(name="Suraj", email="suraj_schema@example.com", password_hash=hash_password("Pass123!"), role="user")
        self.db.add(user)
        self.db.commit()

        schema = UserResponse.model_validate(user)
        dumped = schema.model_dump()
        self.assertNotIn("password_hash", dumped)
        self.assertEqual(dumped["name"], "Suraj")
        self.assertEqual(dumped["email"], "suraj_schema@example.com")
