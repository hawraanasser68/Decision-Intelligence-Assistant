"""routers/health.py

GET /health — liveness check.
"""

from fastapi import APIRouter

from routers import state

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status":    "ok",
        "ml_model":  state._ml_artifact is not None,
        "retriever": state._retriever   is not None,
    }
