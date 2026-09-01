# ==============================================================================
# Migration  : 001_initial_schema.py
# Objective  : Create initial base tables: users, products, orders, and order_items.
# Concept    : Initial Schema Revision
# Why Used   : Establishes base relational schema in migration history.
# ==============================================================================

"""create initial tables

Revision ID: 001
Revises: None
Create Date: 2026-09-01 10:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email')
    )

    # Create products table
    op.create_table(
        'products',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('stock', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('description', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create orders table
    op.create_table(
        'orders',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='PENDING'),
        sa.Column('total_amount', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create order_items table
    op.create_table(
        'order_items',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('order_id', sa.Integer(), nullable=False),
        sa.Column('product_id', sa.Integer(), nullable=False),
        sa.Column('quantity', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('price', sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(['order_id'], ['orders.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['product_id'], ['products.id']),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade() -> None:
    op.drop_table('order_items')
    op.drop_table('orders')
    op.drop_table('products')
    op.drop_table('users')
