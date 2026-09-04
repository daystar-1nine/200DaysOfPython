"""
===============================================================================
DAY 50 — ADMIN ROUTER MODULE (RBAC PROTECTED)
===============================================================================
This module defines administration endpoints for auditing users (GET /admin/users)
and tasks across all accounts (GET /admin/tasks). Protected by require_admin dependency.
===============================================================================
"""

from typing import List
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.schemas.user import UserResponse
from app.schemas.task import TaskResponse
from app.repositories.user_repository import UserRepository
from app.repositories.task_repository import TaskRepository
from app.services.user_service import UserService
from app.services.task_service import TaskService
from app.dependencies.auth import require_admin

router = APIRouter(prefix="/admin", tags=["Admin Operations"])


@router.get("/users", response_model=List[UserResponse], status_code=status.HTTP_200_OK)
def get_all_users_admin(
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> List[UserResponse]:
    """List all registered users (Administrator privilege required)."""
    # What is used: UserService list_users execution protected by require_admin.
    # Why it is used: Provides complete listing of registered accounts for administrative monitoring.
    # How it works: Requires current_user.role == 'admin' or raises 403 Forbidden.
    service = UserService(UserRepository(db))
    users = service.list_users()
    return [UserResponse.model_validate(u) for u in users]


@router.get("/tasks", response_model=List[TaskResponse], status_code=status.HTTP_200_OK)
def get_all_tasks_admin(
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> List[TaskResponse]:
    """List all tasks across all users (Administrator privilege required)."""
    # What is used: TaskService list_all_admin_tasks execution.
    # Why it is used: Allows administrative auditing of system-wide task items.
    # How it works: Validates admin role dependency and returns task list.
    service = TaskService(TaskRepository(db))
    tasks = service.list_all_admin_tasks(offset=offset, limit=limit)
    return [TaskResponse.model_validate(t) for t in tasks]
