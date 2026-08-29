# ==============================================================================
# Test Suite : Database Relationships & Schema Unit Tests (test_relationships.py)
# Objective  : Direct unit testing of SQLAlchemy relationship back_populates, eager loading, and cascading deletes.
# Concept    : SQLAlchemy Relationship Mechanics & Unit Asserts
# Why Used   : Asserts ORM relationship navigation and cascading integrity.
# ==============================================================================

import os
import sys
import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.database import Base
from app.models.user import User
from app.models.product import Product
from app.models.order import Order
from app.models.order_item import OrderItem
from app.repositories.user_repository import UserRepository
from app.schemas.user import UserWithOrdersResponse

class TestDatabaseRelationships(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)

    def test_user_orders_relationship_eager_loading(self):
        user = User(name="Suraj", email="suraj@example.com")
        self.db.add(user)
        self.db.commit()

        order1 = Order(user_id=user.id, total_amount=500.0)
        order2 = Order(user_id=user.id, total_amount=1200.0)
        self.db.add_all([order1, order2])
        self.db.commit()

        # Test eager loading repository method using selectinload
        repo = UserRepository(self.db)
        fetched_user = repo.get_by_id_with_orders(user.id)
        self.assertIsNotNone(fetched_user)
        self.assertEqual(len(fetched_user.orders), 2)
        self.assertEqual(fetched_user.orders[0].total_amount, 500.0)

    def test_user_deletion_cascades_to_orders(self):
        user = User(name="ToDelete", email="delete@example.com")
        self.db.add(user)
        self.db.commit()

        order = Order(user_id=user.id, total_amount=300.0)
        self.db.add(order)
        self.db.commit()

        order_id = order.id
        self.db.delete(user)
        self.db.commit()

        # Verify cascading delete removed order automatically
        self.assertIsNone(self.db.get(Order, order_id))

    def test_order_items_product_relationship(self):
        user = User(name="Alex", email="alex@example.com")
        prod = Product(name="Headphones", price=150.0, stock=5)
        self.db.add_all([user, prod])
        self.db.commit()

        order = Order(user_id=user.id, total_amount=150.0)
        self.db.add(order)
        self.db.flush()

        item = OrderItem(order_id=order.id, product_id=prod.id, quantity=1, price=150.0)
        self.db.add(item)
        self.db.commit()

        self.assertEqual(item.product.name, "Headphones")
        self.assertEqual(item.order.user_id, user.id)

    def test_nested_pydantic_schema_serialization(self):
        user = User(name="Suraj", email="suraj@example.com")
        self.db.add(user)
        self.db.commit()

        order = Order(user_id=user.id, total_amount=500.0)
        self.db.add(order)
        self.db.commit()

        repo = UserRepository(self.db)
        fetched = repo.get_by_id_with_orders(user.id)
        schema_obj = UserWithOrdersResponse.model_validate(fetched)
        self.assertEqual(schema_obj.id, user.id)
        self.assertEqual(len(schema_obj.orders), 1)

    def test_order_status_default(self):
        user = User(name="Suraj", email="suraj@example.com")
        self.db.add(user)
        self.db.commit()

        order = Order(user_id=user.id, total_amount=100.0)
        self.db.add(order)
        self.db.commit()

        self.assertEqual(order.status, "PENDING")

if __name__ == "__main__":
    unittest.main()
