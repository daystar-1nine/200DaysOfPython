# ==============================================================================
# Program    : Health & Observability Router (health.py)
# Objective  : Implement GET /health (Liveness Probe) and GET /health/ready (Readiness Probe).
# Concept    : Observability & Production Health Checking
# Why Used   : Allows load balancers and container orchestrators (Kubernetes/Docker) to probe status.
# ==============================================================================

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from app.database import check_database_readiness
from app.config import settings

router = APIRouter(prefix="/health", tags=["Health & Observability"])

@router.get(
    "",
    status_code=status.HTTP_200_OK,
    summary="Liveness Probe",
    description="Returns 200 OK if application process is running and alive."
)
def liveness_probe():
    """Liveness probe verifying that application process is active."""
    return {
        "status": "alive",
        "app_name": "Production-Ready FastAPI Backend V5",
        "environment": settings.ENVIRONMENT
    }

@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness Probe",
    description="Returns 200 OK if application and database connection are ready to serve traffic."
)
def readiness_probe():
    """Readiness probe checking database connectivity via lightweight query (SELECT 1)."""
    db_ready = check_database_readiness()
    if db_ready:
        return {
            "status": "ready",
            "database": "connected",
            "environment": settings.ENVIRONMENT
        }
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "database": "disconnected",
                "detail": "Database engine is unreachable."
            }
        )
