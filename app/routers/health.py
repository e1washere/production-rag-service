"""Health check router with proper serializable responses."""

import os
import time

from fastapi import APIRouter
from starlette.responses import JSONResponse

from pathlib import Path
from app.config import settings
from app.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health")
def health():
    """Health check endpoint that returns only serializable primitives."""
    try:
        # Basic health info using only primitives
        health_data = {
            "status": "healthy",
            "version": os.getenv("APP_VERSION", "0.1.0"),
            "timestamp": time.time(),
            "environment": settings.environment,
            "checks": {
                "app": True,
                "config": True,
            },
        }

        # Safe check for index directory
        try:
            index_dir = Path(settings.index_dir) if settings.index_dir else None
            if index_dir and index_dir.exists():
                index_files = list(index_dir.glob("*.index"))
                health_data["checks"]["index_dir_exists"] = True
                health_data["checks"]["index_files_found"] = len(index_files) > 0
                health_data["checks"]["index_file_count"] = len(index_files)
            else:
                health_data["checks"]["index_dir_exists"] = False
                health_data["checks"]["index_files_found"] = False
                health_data["checks"]["index_file_count"] = 0
        except Exception as index_error:
            logger.warning(f"Index check failed: {index_error}")
            health_data["checks"]["index_dir_exists"] = False
            health_data["checks"]["index_files_found"] = False
            health_data["checks"]["index_file_count"] = 0

        return JSONResponse(health_data)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time(),
                "checks": {"app": False},
            },
            status_code=503,
        )


@router.get("/ready")
def readiness():
    """Readiness probe for Kubernetes/Container Apps."""
    try:
        # Check if we have the basic requirements
        try:
            index_dir = Path(settings.index_dir) if settings.index_dir else None
            ready = bool(
                index_dir
                and index_dir.exists()
                and len(list(index_dir.glob("*.index"))) > 0
            )
        except Exception:
            ready = False

        if ready:
            return JSONResponse({"status": "ready", "timestamp": time.time()})
        else:
            return JSONResponse(
                {
                    "status": "not_ready",
                    "reason": "index_not_available",
                    "timestamp": time.time(),
                },
                status_code=503,
            )
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            {"status": "not_ready", "error": str(e), "timestamp": time.time()},
            status_code=503,
        )
