"""llm_answer.py

Generates a customer support reply using only the LLM's parametric
knowledge — no retrieved context is injected.

This is the non-RAG baseline for the Generation comparison defined in
the project brief.  Its output is compared directly against the RAG
answer to show what the LLM can and cannot do on its own.
"""

import os
import time
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHAT_MODEL = "gpt-4o-mini"

# GPT-4o-mini pricing (USD per token, as of 2025)
COST_PER_INPUT_TOKEN  = 0.150 / 1_000_000
COST_PER_OUTPUT_TOKEN = 0.600 / 1_000_000

SYSTEM_PROMPT = (
    "You are a helpful customer support agent. "
    "A customer has sent you the following support ticket. "
    "Write a clear, empathetic, and actionable reply. "
    "Be concise (2–4 sentences). "
    "Do not ask for information you do not need. "
    "Do not mention that you are an AI."
)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def generate_answer_no_rag(
    query: str,
    api_key: str | None = None,
    model: str = CHAT_MODEL,
) -> dict:
    """Generate a support reply with the LLM only — no retrieved context.

    Args:
        query:   Raw customer ticket text.
        api_key: OpenAI API key.  Defaults to the OPENAI_API_KEY env var.
        model:   Chat model identifier.

    Returns:
        dict with keys:
            answer     (str)   — generated reply text
            latency_ms (float) — wall-clock time for the API call
            cost_usd   (float) — per-call cost in USD
            model_name (str)   — model identifier
    """
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": query},
        ],
        temperature=0.3,
        max_tokens=256,
    )
    latency_ms = (time.perf_counter() - t0) * 1_000

    answer = response.choices[0].message.content or ""
    usage  = response.usage
    cost   = (
        usage.prompt_tokens     * COST_PER_INPUT_TOKEN
        + usage.completion_tokens * COST_PER_OUTPUT_TOKEN
    )

    return {
        "answer":     answer.strip(),
        "latency_ms": round(latency_ms, 1),
        "cost_usd":   round(cost, 8),
        "model_name": model,
    }
