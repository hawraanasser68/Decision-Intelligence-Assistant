"""build_chroma_index.py

Reads the RAG corpus CSV, generates embeddings for each customer question
using the OpenAI embeddings API, and stores them in a persistent Chroma
vector database.

This script only needs to be run once (or when the corpus changes).
The resulting Chroma index is stored at artifacts/chroma_db/ and loaded
at runtime by the backend retriever.

Why embed only the question and not the answer?
  At retrieval time we have a new user query — a customer question. We want
  to find past questions that are semantically similar to it. The answer is
  returned as metadata alongside the matched question, but is not part of
  the similarity search itself.

Usage:
    python scripts/build_chroma_index.py \\
        --corpus  data/processed/rag_corpus.csv \\
        --db-path artifacts/chroma_db \\
        --batch-size 512

Environment variables (set in .env):
    OPENAI_API_KEY — required
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Ensure project root is on path for local imports.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from chromadb.config import Settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "support_tickets"

# text-embedding-3-small pricing: $0.020 per 1,000,000 tokens
EMBED_COST_PER_TOKEN = 0.020 / 1_000_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_embeddings(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    """Call the OpenAI embeddings API for a batch of texts.

    Args:
        client: Authenticated OpenAI client.
        texts:  List of strings to embed. Max 2048 items per call.
        model:  Embedding model name.

    Returns:
        List of embedding vectors in the same order as `texts`.
    """
    response = client.embeddings.create(input=texts, model=model)
    # The API returns embeddings sorted by index, so the order is safe.
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def build_index(corpus_csv: Path, db_path: Path, batch_size: int) -> None:
    """Embed the corpus questions and store them in a persistent Chroma collection.

    Args:
        corpus_csv:  Path to rag_corpus.csv produced by build_rag_corpus.py.
        db_path:     Directory where Chroma will persist the index.
        batch_size:  Number of rows to embed per API call (max 2048).
    """
    # --- Load env and validate API key ---
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set in environment or .env file.")

    client = OpenAI()

    # --- Load corpus ---
    logger.info("Loading corpus from %s", corpus_csv)
    df = pd.read_csv(corpus_csv)
    logger.info("Total QA pairs: %d", len(df))

    # Drop rows with empty questions just in case.
    df = df[df["question"].str.strip().str.len() > 0].reset_index(drop=True)
    logger.info("Rows with non-empty questions: %d", len(df))

    # --- Set up Chroma ---
    db_path.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(db_path))

    # Delete existing collection so the index is always rebuilt cleanly.
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing collection '%s'", COLLECTION_NAME)
    except Exception:
        pass  # Collection did not exist yet — that's fine.

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        # Cosine distance is standard for embedding similarity.
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Created Chroma collection '%s'", COLLECTION_NAME)

    # --- Embed and insert in batches ---
    total_rows = len(df)
    total_tokens = 0
    start_time = time.time()

    for batch_start in range(0, total_rows, batch_size):
        batch = df.iloc[batch_start : batch_start + batch_size]

        questions = batch["question"].tolist()
        ids       = batch["tweet_id"].astype(str).tolist()

        # Metadata stored alongside each vector — returned at retrieval time.
        metadatas = [
            {
                "question":           row["question"],
                "answer":             str(row["answer"])[:2000],  # cap length for Chroma metadata limit
                "priority_label":     int(row["priority_label"]),
                "priority_label_str": str(row["priority_label_str"]),
                "weak_score":         float(row["weak_score"]),
                "sentiment_score":    float(row["sentiment_score"]),
            }
            for _, row in batch.iterrows()
        ]

        embeddings = get_embeddings(client, questions, EMBEDDING_MODEL)

        # Track approximate token cost.
        # Rough estimate: 1 token ≈ 4 characters.
        batch_tokens = sum(len(q) // 4 for q in questions)
        total_tokens += batch_tokens

        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "Indexed %d / %d  (~$%.4f so far)",
            min(batch_start + batch_size, total_rows),
            total_rows,
            total_tokens * EMBED_COST_PER_TOKEN,
        )

    elapsed = time.time() - start_time
    estimated_cost = total_tokens * EMBED_COST_PER_TOKEN

    logger.info("Indexing complete in %.1f s", elapsed)
    logger.info("Estimated embedding cost: $%.4f", estimated_cost)
    logger.info("Chroma index saved to %s", db_path)
    logger.info("Collection size: %d vectors", collection.count())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed the RAG corpus and store it in a Chroma vector DB."
    )
    parser.add_argument(
        "--corpus",
        default="data/processed/rag_corpus_train.csv",
        help="Path to the train split of the corpus (default: data/processed/rag_corpus_train.csv)",
    )
    parser.add_argument(
        "--db-path",
        default="artifacts/chroma_db",
        help="Directory to persist the Chroma index (default: artifacts/chroma_db)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Rows per OpenAI embeddings API call (default: 512, max 2048)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_index(
        corpus_csv=Path(args.corpus),
        db_path=Path(args.db_path),
        batch_size=args.batch_size,
    )
