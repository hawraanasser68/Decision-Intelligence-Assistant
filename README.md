# Decision Intelligence Assistant

A full-stack customer support triage system that compares **four outputs** for every query:

| Output | Description |
|---|---|
| **RAG Answer** | LLM reply grounded in retrieved past cases from the vector store |
| **Non-RAG Answer** | LLM reply with no context — parametric knowledge only |
| **ML Priority** | Random Forest classifier (fast, free, ~2 ms) |
| **LLM Zero-Shot Priority** | GPT-4o-mini asked "is this urgent?" directly |

---

## Stack

- **Backend** — FastAPI + ChromaDB (in-process, persistent) + OpenAI + scikit-learn
- **Frontend** — React (Vite) + nginx
- **Embeddings** — `text-embedding-3-small`
- **LLM** — `gpt-4o-mini`
- **Vector store** — Chroma (persistent mode, no separate container needed)

---

## Running the stack

### Prerequisites

- Docker + Docker Compose installed
- An OpenAI API key

### 1. Set up environment

```bash
cp .env.example .env
# Edit .env and fill in your OPENAI_API_KEY
```

### 2. Start everything

```bash
docker compose up --build
```

The first build takes ~2–3 minutes (downloads Python and Node images, installs packages).

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |

### 3. Stop

```bash
docker compose down
```

### 4. Rebuild from scratch (wipe volumes)

```bash
docker compose down -v
docker compose up --build
```

---

## Running locally (without Docker)

```bash
# Create and activate the virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install backend dependencies
pip install fastapi uvicorn openai pydantic python-dotenv pandas scikit-learn joblib nltk chromadb
python -c "import nltk; nltk.download('vader_lexicon')"

# Start the backend
uvicorn main:app --reload --port 8000

# In a second terminal — start the frontend
cd frontend
npm install
npm run dev        # http://localhost:5173
```

---

## Building the data pipeline (one-time)

```bash
# 1. Build the RAG corpus from the raw Twitter dataset
python scripts/build_rag_corpus.py \
  --input  data/processed/twcs.csv \
  --output data/processed/rag_corpus_train.csv

# 2. Embed and index into Chroma (~40 min, costs ~$0.80 in embeddings)
python scripts/build_chroma_index.py \
  --corpus   data/processed/rag_corpus_train.csv \
  --db-path  artifacts/chroma_db \
  --batch-size 512
```

---

## API reference

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check — shows whether ML model and retriever are loaded |
| POST | `/compare` | **Main endpoint** — returns all four outputs for one ticket |
| POST | `/predict/ml` | ML Random Forest priority only |
| POST | `/predict/llm` | LLM zero-shot priority only |
| POST | `/answer/rag` | RAG answer + retrieved cases |
| POST | `/answer/non-rag` | Non-RAG LLM answer |

All POST endpoints accept `{ "text": "customer ticket text" }`.

---

## Logging

Every query is appended to `logs/query_log.jsonl` (persisted via Docker volume).  
Each entry includes: timestamp, endpoint, input text, outputs, latency (ms), and cost (USD).

---

## Vector store justification

Chroma runs **in-process** in persistent mode (`artifacts/chroma_db/`).  
No separate `vector-db` container is needed — the backend mounts the data directory as a named volume.  
At 641,995 vectors this is well within Chroma's single-node capacity, and latency is ~100–300 ms per query (embedding + ANN search).

---

## Tradeoff analysis: ML vs LLM priority prediction

| Dimension | ML (Random Forest) | LLM Zero-Shot |
|---|---|---|
| Accuracy | ~87% (test set) | ~91% (test set) |
| Latency | ~2 ms | ~800 ms |
| Cost per call | $0 | ~$0.000 02 |
| Cost at 10k/hr | $0 | ~$0.20/hr |

**Recommendation:** Deploy the ML model for real-time triage at scale. Use the LLM only when reasoning or explainability matters. At 10,000 tickets/hour the ML model costs nothing and returns in microseconds; the LLM costs ~$0.20/hour and adds ~800 ms latency per ticket.
