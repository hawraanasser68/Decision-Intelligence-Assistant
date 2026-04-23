"""routers/predict.py

POST /predict/ml  — Random Forest priority classification.
POST /predict/llm — LLM zero-shot priority classification.
"""

from datetime import datetime

from fastapi import APIRouter

from routers import state
from schemas.inference import PredictionResponse, TicketRequest

router = APIRouter(prefix="/predict")


@router.post("/ml", response_model=PredictionResponse)
def predict_ml(request: TicketRequest):
    """Random Forest priority classification."""
    result = state.ml_predict(request.text)
    state.log_query({
        "ts": datetime.utcnow().isoformat(), "endpoint": "predict/ml",
        "text": request.text, "label": result.label.value,
        "confidence": result.confidence, "latency_ms": result.latency_ms, "cost_usd": 0.0,
    })
    return PredictionResponse(text=request.text, source="ml", prediction=result)


@router.post("/llm", response_model=PredictionResponse)
def predict_llm(request: TicketRequest):
    """LLM zero-shot priority classification."""
    result = state.llm_predict(request.text)
    state.log_query({
        "ts": datetime.utcnow().isoformat(), "endpoint": "predict/llm",
        "text": request.text, "label": result.label.value,
        "confidence": result.confidence, "latency_ms": result.latency_ms,
        "cost_usd": result.cost_usd,
    })
    return PredictionResponse(text=request.text, source="llm", prediction=result)
