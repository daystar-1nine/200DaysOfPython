# ==============================================================================
# Migration  : 006_add_user_created_at.py
# Objective  : Add created_at column to users with default data population for existing records.
# Concept    : Data Migration & Existing Data Handling (Bonus Challenge)
# Why Used   : Ensures existing user records receive sensible creation timestamps.
# ==============================================================================

"""add user created_at with data migration

Revision ID: 006
Revises: 005
Create Date: 2026-09-01 10:50:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from datetime import datetime, timezone

revision: str = '006'
down_revision: Union[str, None] = '005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Step 1: Add column allowing NULL initially
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(sa.Column('created_at', sa.DateTime(), nullable=True))

    # Step 2: Data Migration — Populate default timestamp for existing rows
    users_table = sa.table('users', sa.column('created_at', sa.DateTime()))
    op.execute(
        users_table.update().where(users_table.c.created_at.is_(None)).values(
            created_at=datetime.now(timezone.utc)
        )
    )

def downgrade() -> None:
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_column('created_at')
