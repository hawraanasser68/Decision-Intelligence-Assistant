"""main.py

FastAPI backend for the Decision Intelligence Assistant.

Endpoints are defined in the routers/ package:
  routers/health.py   — GET  /health
  routers/predict.py  — POST /predict/ml, POST /predict/llm
  routers/answer.py   — POST /answer/non-rag, POST /answer/rag
  routers/compare.py  — POST /compare

Shared application state (ML model, Retriever, OpenAI client) lives in
routers/state.py and is populated once at startup via the lifespan below.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from rag.retriever import Retriever
from routers import answer, compare, health, predict
from routers import state

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — load heavy resources once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = PROJECT_ROOT / "artifacts" / "priority_model.joblib"
    if not model_path.exists():
        logger.warning("ML model not found at %s — /predict/ml will fail", model_path)
    else:
        state._ml_artifact = joblib.load(model_path)
        logger.info("ML model loaded (%s)", model_path.name)

    db_path = str(PROJECT_ROOT / "artifacts" / "chroma_db")
    try:
        state._retriever = Retriever(db_path=db_path)
        logger.info("Retriever loaded from %s", db_path)
    except Exception as exc:
        logger.warning("Retriever not available (%s) — /answer/rag will fail", exc)

    state._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    yield

    logger.info("Shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Decision Intelligence Assistant",
    description="RAG + LLM + ML priority prediction for customer support tickets",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(answer.router)
app.include_router(compare.router)
