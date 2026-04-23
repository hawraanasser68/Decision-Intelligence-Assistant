"""routers/compare.py

POST /compare — All four outputs in one call (the main UI endpoint).
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException

from rag.llm_answer import generate_answer_no_rag
from rag.rag_answer import generate_answer_with_rag
from routers import state
from schemas.inference import AnswerResult, CompareAllResponse, TicketRequest

router = APIRouter()


@router.post("/compare", response_model=CompareAllResponse)
def compare(request: TicketRequest):
    """All four outputs for one ticket: RAG answer, non-RAG answer,
    ML priority prediction, LLM zero-shot priority prediction."""
    if state._retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not loaded")

    ml_result  = state.ml_predict(request.text)
    llm_result = state.llm_predict(request.text)

    non_rag_result = AnswerResult(**generate_answer_no_rag(request.text))

    rag_dict, cases, ret_ms = generate_answer_with_rag(request.text, state._retriever)
    rag_result   = AnswerResult(**rag_dict)
    cases_schema = state.cases_to_schema(cases)

    state.log_query({
        "ts": datetime.utcnow().isoformat(), "endpoint": "compare",
        "text": request.text,
        "ml_label": ml_result.label.value,   "ml_confidence": ml_result.confidence,
        "ml_latency_ms": ml_result.latency_ms,
        "llm_label": llm_result.label.value,  "llm_confidence": llm_result.confidence,
        "llm_latency_ms": llm_result.latency_ms, "llm_cost_usd": llm_result.cost_usd,
        "non_rag_answer": non_rag_result.answer,
        "non_rag_latency_ms": non_rag_result.latency_ms, "non_rag_cost_usd": non_rag_result.cost_usd,
        "rag_answer": rag_result.answer,
        "rag_latency_ms": rag_result.latency_ms, "rag_cost_usd": rag_result.cost_usd,
        "retrieval_latency_ms": ret_ms,
        "retrieved_cases": [{"tweet_id": c.tweet_id, "similarity": c.similarity} for c in cases_schema],
    })

    return CompareAllResponse(
        text=request.text,
        rag_answer=rag_result,
        non_rag_answer=non_rag_result,
        ml_prediction=ml_result,
        llm_prediction=llm_result,
        retrieved_cases=cases_schema,
        retrieval_latency_ms=ret_ms,
    )
