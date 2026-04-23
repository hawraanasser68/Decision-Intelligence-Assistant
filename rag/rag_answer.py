"""rag_answer.py

Generates a customer support reply using RAG: the top-k most similar
past cases are retrieved from the Chroma vector store and injected into
the prompt as few-shot examples before asking the LLM to compose a reply.

The generated answer is grounded in real past resolutions and should be
more specific and accurate than the plain-LLM baseline in llm_answer.py.
"""

import os
import time

from openai import OpenAI

from rag.retriever import Retriever, RetrievedCase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHAT_MODEL = "gpt-4o-mini"

COST_PER_INPUT_TOKEN  = 0.150 / 1_000_000
COST_PER_OUTPUT_TOKEN = 0.600 / 1_000_000

SYSTEM_PROMPT = (
    "You are a helpful customer support agent. "
    "You will be given a customer query and a set of similar resolved cases "
    "from our support history. "
    "Base your reply ONLY on the information present in those cases — "
    "do not add any facts, policies, timeframes, or details that are not "
    "explicitly mentioned in the provided cases. "
    "If the cases do not contain enough information to give a specific answer, "
    "say so and ask the customer to DM their order details. "
    "Write a clear, empathetic, and actionable reply in 2–4 sentences. "
    "Do not mention that you are an AI."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_cases(cases: list[RetrievedCase]) -> str:
    """Render retrieved cases as numbered examples for the prompt."""
    lines = []
    for i, case in enumerate(cases, start=1):
        lines.append(
            f"Case {i} (similarity {case.similarity:.2f}, "
            f"priority: {case.priority_label_str}):\n"
            f"  Customer: {case.question}\n"
            f"  Agent reply: {case.answer}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def generate_answer_with_rag(
    query: str,
    retriever: Retriever,
    top_k: int = 5,
    api_key: str | None = None,
    model: str = CHAT_MODEL,
) -> tuple[dict, list[RetrievedCase], float]:
    """Generate a support reply grounded in retrieved past cases.

    Args:
        query:      Raw customer ticket text.
        retriever:  A loaded Retriever instance (shared across requests).
        top_k:      Number of past cases to retrieve.
        api_key:    OpenAI API key.  Defaults to the OPENAI_API_KEY env var.
        model:      Chat model identifier.

    Returns:
        Tuple of:
          - dict with keys: answer, latency_ms, cost_usd, model_name
          - list of RetrievedCase objects that were used as context
          - retrieval latency in milliseconds (embedding + Chroma query)
    """
    # Step 1 — retrieve similar past cases.
    cases, retrieval_latency_ms = retriever.retrieve(query, top_k=top_k)

    # Step 2 — build the prompt with retrieved cases as context.
    context_block = _format_cases(cases)
    user_message = (
        f"Similar resolved cases from our support history:\n\n"
        f"{context_block}\n\n"
        f"---\n\n"
        f"New customer query:\n{query}"
    )

    # Step 3 — call the LLM.
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.3,
        max_tokens=256,
    )
    generation_latency_ms = (time.perf_counter() - t0) * 1_000

    answer = response.choices[0].message.content or ""
    usage  = response.usage
    cost   = (
        usage.prompt_tokens     * COST_PER_INPUT_TOKEN
        + usage.completion_tokens * COST_PER_OUTPUT_TOKEN
    )

    answer_dict = {
        "answer":     answer.strip(),
        "latency_ms": round(generation_latency_ms, 1),
        "cost_usd":   round(cost, 8),
        "model_name": model,
    }

    return answer_dict, cases, round(retrieval_latency_ms, 1)
