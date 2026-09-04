"""
===============================================================================
DAY 50 — HEALTH & OBSERVABILITY PROBES ROUTER
===============================================================================
This module provides Liveness (GET /health) and Database Readiness (GET /health/ready)
health check endpoints for container orchestrators.
===============================================================================
"""

from typing import Dict
from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.database import get_db

router = APIRouter(tags=["Health & Observability"])


@router.get("/health", status_code=status.HTTP_200_OK)
def health_liveness() -> Dict[str, str]:
    """Liveness probe indicating application server process is running."""
    # What is used: Simple status payload response.
    # Why it is used: Informs load balancers that application process is alive.
    # How it works: Returns static status ok dictionary.
    return {"status": "ok", "service": "TaskFlow API"}


@router.get("/health/ready", status_code=status.HTTP_200_OK)
def health_readiness(db: Session = Depends(get_db)) -> Dict[str, str]:
    """Readiness probe testing database connection via SQL query execution."""
    # What is used: Raw SQL execution `SELECT 1`.
    # Why it is used: Verifies active database connectivity before serving traffic.
    # How it works: Executes SELECT 1 on database connection; returns healthy status.
    db.execute(text("SELECT 1"))
    return {"status": "ready", "database": "connected"}
