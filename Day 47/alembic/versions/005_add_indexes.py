"""add indexes on users email and products name

Revision ID: 005
Revises: 004
Create Date: 2026-09-01 10:20:00.000000

"""
from typing import Sequence, Union
from alembic import op

revision: str = '005'
down_revision: Union[str, None] = '004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.create_index('idx_users_email', ['email'], unique=True)

    with op.batch_alter_table('products', schema=None) as batch_op:
        batch_op.create_index('idx_products_name', ['name'], unique=False)

def downgrade() -> None:
    with op.batch_alter_table('products', schema=None) as batch_op:
        batch_op.drop_index('idx_products_name')

    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_index('idx_users_email')
