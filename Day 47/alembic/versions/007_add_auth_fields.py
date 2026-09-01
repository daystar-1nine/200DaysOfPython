"""add authentication fields password_hash, role, and age to users table

Revision ID: 007
Revises: 006
Create Date: 2026-09-01 12:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '007'
down_revision: Union[str, None] = '006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(sa.Column('password_hash', sa.String(length=255), server_default='default_hash_for_existing', nullable=False))
        batch_op.add_column(sa.Column('role', sa.String(length=50), server_default='user', nullable=False))
        batch_op.add_column(sa.Column('age', sa.Integer(), nullable=True))

def downgrade() -> None:
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_column('age')
        batch_op.drop_column('role')
        batch_op.drop_column('password_hash')
