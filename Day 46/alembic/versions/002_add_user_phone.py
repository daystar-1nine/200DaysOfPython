# ==============================================================================
# Migration  : 002_add_user_phone.py
# Objective  : Add optional phone column to users table.
# Concept    : Incremental Column Addition
# Why Used   : Modifies users table schema safely without dropping existing user data.
# ==============================================================================

"""add user phone column

Revision ID: 002
Revises: 001
Create Date: 2026-09-01 10:10:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(sa.Column('phone', sa.String(length=20), nullable=True))

def downgrade() -> None:
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_column('phone')
