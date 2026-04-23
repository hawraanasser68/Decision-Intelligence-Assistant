"""routers/state.py

Shared application state and helper functions used by all routers.

The lifespan in main.py populates _ml_artifact, _retriever, and _openai
at startup. Every router imports this module and reads from it directly.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException
from openai import OpenAI

from rag.retriever import Retriever
from schemas.inference import PredictionResult, RetrievedCaseSchema
from schemas.priority import Priority, TicketPriority
from utils.ml_features import extract_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM constants
# ---------------------------------------------------------------------------
LLM_MODEL             = "gpt-4o-mini"
COST_PER_INPUT_TOKEN  = 0.150 / 1_000_000
COST_PER_OUTPUT_TOKEN = 0.600 / 1_000_000

TRIAGE_SYSTEM_PROMPT = (
    "You are a customer support triage assistant. "
    "Your job is to classify support tickets as either 'urgent' or 'normal'.\n\n"
    "Urgent tickets involve: service outages, billing emergencies, data loss, "
    "safety concerns, or extreme customer distress.\n"
    "Normal tickets involve: general questions, minor issues, feature requests, "
    "or routine account inquiries.\n\n"
    "Respond with your classification label, a confidence score between 0.0 and 1.0, "
    "and a brief one-sentence reasoning."
)

# ---------------------------------------------------------------------------
# Application state — set by lifespan in main.py
# ---------------------------------------------------------------------------
_ml_artifact: dict | None    = None
_retriever:   Retriever | None = None
_openai:      OpenAI | None   = None

# ---------------------------------------------------------------------------
# Query logging
# ---------------------------------------------------------------------------
_LOGS_DIR = Path(__file__).resolve().parents[1] / "logs"
_LOGS_DIR.mkdir(exist_ok=True)
QUERY_LOG = _LOGS_DIR / "query_log.jsonl"


def log_query(entry: dict) -> None:
    try:
        with QUERY_LOG.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning("Failed to write query log: %s", exc)


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------

def ml_predict(text: str) -> PredictionResult:
    if _ml_artifact is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")

    pipeline    = _ml_artifact["pipeline"]
    features_df = extract_features(text)

    t0         = time.perf_counter()
    pred_label = pipeline.predict(features_df)[0]
    proba      = pipeline.predict_proba(features_df)[0]
    latency_ms = (time.perf_counter() - t0) * 1_000

    label = Priority.urgent if int(pred_label) == 1 else Priority.normal
    return PredictionResult(
        label=label,
        confidence=round(float(max(proba)), 4),
        reasoning=None,
        model_name="random_forest",
        latency_ms=round(latency_ms, 2),
        cost_usd=0.0,
    )


def llm_predict(text: str) -> PredictionResult:
    if _openai is None:
        raise HTTPException(status_code=503, detail="OpenAI client not initialised")

    t0 = time.perf_counter()
    response = _openai.beta.chat.completions.parse(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ],
        response_format=TicketPriority,
        temperature=0.0,
    )
    latency_ms = (time.perf_counter() - t0) * 1_000

    parsed = response.choices[0].message.parsed
    usage  = response.usage
    cost   = (
        usage.prompt_tokens     * COST_PER_INPUT_TOKEN
        + usage.completion_tokens * COST_PER_OUTPUT_TOKEN
    )
    return PredictionResult(
        label=parsed.label,
        confidence=round(parsed.confidence, 4),
        reasoning=parsed.reasoning,
        model_name=LLM_MODEL,
        latency_ms=round(latency_ms, 2),
        cost_usd=round(cost, 8),
    )


def cases_to_schema(cases) -> list[RetrievedCaseSchema]:
    return [
        RetrievedCaseSchema(
            tweet_id=c.tweet_id,
            question=c.question,
            answer=c.answer,
            priority_label_str=c.priority_label_str,
            similarity=round(c.similarity, 4),
        )
        for c in cases
    ]
