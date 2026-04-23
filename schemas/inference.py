"""Reusable request and response schemas for application inference flows.

These models are intended for the FastAPI layer and for any code that needs
a consistent contract for ticket submission, single-model predictions,
generated answers, or the full four-way comparison required by the brief.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from schemas.priority import Priority


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class TicketRequest(BaseModel):
    """Request body: raw customer ticket text."""
    text: str = Field(..., min_length=1, description="Customer support ticket text")


# ---------------------------------------------------------------------------
# Priority prediction (ML and LLM zero-shot)
# ---------------------------------------------------------------------------

class PredictionResult(BaseModel):
    """Normalized priority prediction shared across ML and LLM outputs."""

    label: Priority

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0",
    )

    reasoning: str | None = Field(
        default=None,
        description="Short explanation for the prediction",
    )

    model_name: str | None = Field(
        default=None,
        description="Model identifier such as random_forest or gpt-4o-mini",
    )

    latency_ms: float | None = Field(
        default=None,
        description="Wall-clock inference time in milliseconds",
    )

    cost_usd: float | None = Field(
        default=None,
        description="Per-call cost in USD (zero for ML models)",
    )


# ---------------------------------------------------------------------------
# Generated answer (RAG and non-RAG)
# ---------------------------------------------------------------------------

class AnswerResult(BaseModel):
    """A generated support reply from the LLM, with or without RAG context."""

    answer: str = Field(..., description="Generated support reply text")

    latency_ms: float = Field(..., description="Wall-clock generation time in milliseconds")

    cost_usd: float = Field(..., description="Per-call cost in USD")

    model_name: str = Field(default="gpt-4o-mini", description="LLM model identifier")


# ---------------------------------------------------------------------------
# Retrieved case (returned to the frontend source panel)
# ---------------------------------------------------------------------------

class RetrievedCaseSchema(BaseModel):
    """One past support case retrieved from the vector store."""

    tweet_id: str
    question: str = Field(..., description="Original customer question")
    answer: str = Field(..., description="Agent reply from the corpus")
    priority_label_str: str = Field(..., description="normal or urgent")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")


# ---------------------------------------------------------------------------
# Single-source responses (used by individual /predict and /answer endpoints)
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    """Response for a single priority prediction endpoint."""
    text: str
    source: str = Field(..., description="ml or llm")
    prediction: PredictionResult


class AnswerResponse(BaseModel):
    """Response for a single answer generation endpoint."""
    text: str
    source: str = Field(..., description="rag or non_rag")
    answer_result: AnswerResult
    retrieved_cases: list[RetrievedCaseSchema] = Field(
        default_factory=list,
        description="Retrieved cases (populated for RAG only)",
    )
    retrieval_latency_ms: float | None = Field(
        default=None,
        description="Retrieval latency in milliseconds (RAG only)",
    )


# ---------------------------------------------------------------------------
# Four-way comparison response (the core deliverable)
# ---------------------------------------------------------------------------

class CompareAllResponse(BaseModel):
    """Full four-way comparison: RAG answer, non-RAG answer, ML priority, LLM priority."""

    text: str

    # Generated answers
    rag_answer: AnswerResult
    non_rag_answer: AnswerResult

    # Priority predictions
    ml_prediction: PredictionResult
    llm_prediction: PredictionResult

    # RAG context shown in source panel
    retrieved_cases: list[RetrievedCaseSchema]
    retrieval_latency_ms: float