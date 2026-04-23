"""routers/answer.py

POST /answer/non-rag — LLM answer generation (no RAG context).
POST /answer/rag     — RAG-augmented LLM answer generation.
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException

from rag.llm_answer import generate_answer_no_rag
from rag.rag_answer import generate_answer_with_rag
from routers import state
from schemas.inference import AnswerResponse, AnswerResult, TicketRequest

router = APIRouter(prefix="/answer")


@router.post("/non-rag", response_model=AnswerResponse)
def answer_no_rag(request: TicketRequest):
    """Generate a support reply using the LLM alone (no retrieval)."""
    result = AnswerResult(**generate_answer_no_rag(request.text))
    state.log_query({
        "ts": datetime.utcnow().isoformat(), "endpoint": "answer/non-rag",
        "text": request.text, "answer": result.answer,
        "latency_ms": result.latency_ms, "cost_usd": result.cost_usd,
    })
    return AnswerResponse(text=request.text, source="non_rag", answer_result=result)


@router.post("/rag", response_model=AnswerResponse)
def answer_rag(request: TicketRequest):
    """Generate a support reply grounded in retrieved past cases."""
    if state._retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not loaded")

    result_dict, cases, ret_ms = generate_answer_with_rag(request.text, state._retriever)
    result       = AnswerResult(**result_dict)
    cases_schema = state.cases_to_schema(cases)

    state.log_query({
        "ts": datetime.utcnow().isoformat(), "endpoint": "answer/rag",
        "text": request.text, "answer": result.answer,
        "latency_ms": result.latency_ms, "retrieval_latency_ms": ret_ms,
        "cost_usd": result.cost_usd,
        "retrieved_cases": [{"tweet_id": c.tweet_id, "similarity": c.similarity} for c in cases_schema],
    })
    return AnswerResponse(
        text=request.text, source="rag", answer_result=result,
        retrieved_cases=cases_schema, retrieval_latency_ms=ret_ms,
    )
