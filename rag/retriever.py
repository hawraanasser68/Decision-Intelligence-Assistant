"""retriever.py

Retrieves the top-k most similar past support cases from the Chroma vector
database for a given user query.

This module is the single retrieval interface used by the FastAPI backend.
It applies the same text cleaning as the corpus build step (via utils.text)
before embedding the query, so the query and corpus vectors are in the same
space.

Usage (from the backend):
    from rag.retriever import Retriever

    retriever = Retriever()          # loads the index once at startup
    results = retriever.retrieve("my account has been hacked", top_k=5)
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on path for local imports.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from openai import OpenAI

from utils.text import clean_text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_MODEL  = "text-embedding-3-small"
COLLECTION_NAME  = "support_tickets"
DEFAULT_DB_PATH  = str(PROJECT_ROOT / "artifacts" / "chroma_db")
DEFAULT_TOP_K    = 5


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetrievedCase:
    """One retrieved support case returned by the retriever."""

    tweet_id:           str
    question:           str    # cleaned customer question from the corpus
    answer:             str    # concatenated company reply thread
    priority_label:     int    # 0 = normal, 1 = urgent
    priority_label_str: str    # "normal" or "urgent"
    weak_score:         float
    sentiment_score:    float
    similarity:         float  # cosine similarity score (0–1, higher = more similar)


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class Retriever:
    """Loads the Chroma index once and exposes a .retrieve() method.

    The index is loaded into memory when the object is created, which
    happens once at backend startup. All subsequent queries reuse the
    same in-memory index — no disk I/O per query.

    Args:
        db_path:    Path to the persisted Chroma directory.
        api_key:    OpenAI API key. Defaults to OPENAI_API_KEY env var.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        api_key: str | None = None,
    ) -> None:
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        chroma_client = chromadb.PersistentClient(path=db_path)
        self._collection = chroma_client.get_collection(COLLECTION_NAME)

    def _embed(self, text: str) -> list[float]:
        """Embed a single cleaned string using the OpenAI embeddings API."""
        response = self._client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL,
        )
        return response.data[0].embedding

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> tuple[list[RetrievedCase], float]:
        """Find the top-k most similar past cases for a raw user query.

        The query is cleaned with the same function used during corpus
        construction before embedding, so the vector spaces align.

        Args:
            query:  Raw user query string.
            top_k:  Number of results to return.

        Returns:
            Tuple of:
              - List of RetrievedCase objects, sorted by similarity descending.
              - Latency in milliseconds for the full retrieval call.
        """
        # Apply the same cleaning used at corpus build time.
        clean_query = clean_text(query)

        t0 = time.perf_counter()
        query_embedding = self._embed(clean_query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        cases = []
        for meta, distance in zip(
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Chroma returns cosine *distance* (0 = identical, 2 = opposite).
            # Convert to similarity: similarity = 1 - (distance / 2)
            similarity = 1.0 - (distance / 2.0)

            cases.append(RetrievedCase(
                tweet_id=meta.get("tweet_id", ""),
                question=meta.get("question", ""),
                answer=meta.get("answer", ""),
                priority_label=int(meta.get("priority_label", 0)),
                priority_label_str=str(meta.get("priority_label_str", "normal")),
                weak_score=float(meta.get("weak_score", 0.0)),
                sentiment_score=float(meta.get("sentiment_score", 0.0)),
                similarity=round(similarity, 4),
            ))

        return cases, round(latency_ms, 2)
