# Cents Backend

Cents is a multi-user RAG chatbot backend built with Python 3.12, FastAPI, LangGraph, SQLite, and ChromaDB.

This repository now uses a single backend stack:
- relational/auth/conversation data in SQLite
- vector storage and similarity search in ChromaDB

PostgreSQL, pgvector, Alembic, Docker Compose, and WSL are not required for normal development.

## High-level architecture

```mermaid
flowchart LR
    A[Frontend] -->|HTTPS + JWT| B[FastAPI API]
    B --> C[Auth layer\nJWT + bcrypt]
    B --> D[Conversations API]
    B --> E[Chat API]
    B --> F[Documents API]
    B --> G[Tools API]

    E --> H[LangGraph orchestrator]
    H --> I[Tool retrieval node]
    H --> J[Document retrieval node]
    H --> K[Generation node]
    K --> L[Judge node]
    L -->|pass| M[Response]
    L -->|retry| H

    C --> N[SQLite\nusers]
    D --> O[SQLite\nconversations]
    F --> P[SQLite\ndocuments metadata]
    G --> Q[SQLite\ntool definitions]

    I --> R[ChromaDB\ntool vectors]
    J --> S[ChromaDB\ndocument vectors]
```

## What this project includes

- FastAPI backend with CORS support
- JWT auth via local auth routes and bcrypt password hashing
- Conversation CRUD (user-scoped)
- SSE chat endpoint backed by LangGraph orchestration
- Document ingestion into SQLite + Chroma
- Tool registration into SQLite + Chroma

## Project structure

```text
app/
├── auth/
│   └── users.py
├── db/
│   ├── base.py
│   └── models.py
├── graph/
│   ├── state.py
│   ├── orchestrator.py
│   ├── retrieval_tools.py
│   ├── retrieval_docs.py
│   ├── generation.py
│   ├── judge.py
│   ├── graph.py
│   └── checkpointer.py
├── routes/
│   ├── conversations.py
│   ├── chat.py
│   ├── documents.py
│   └── tools.py
├── vector_store.py
├── config.py
└── main.py
```

## Python version

Python 3.12 is required.

## Setup

### Windows (native)

From the repository root:

```powershell
scripts\setup.bat
scripts\start.bat
```

Alternative PowerShell scripts:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

### macOS / Linux

```bash
chmod +x scripts/setup.sh scripts/start.sh
./scripts/setup.sh
./scripts/start.sh
```

## Manual setup (all platforms)

On macOS/Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import asyncio; from app.db.base import init_db; from app.vector_store import ensure_vector_store_ready; asyncio.run(init_db()); ensure_vector_store_ready()"
.venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

On Windows native, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.

## Start commands

Windows (CMD):

```bat
scripts\start.bat
```

Windows (PowerShell):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

macOS / Linux:

```bash
./scripts/start.sh
```

## Environment

This project uses one environment file: `.env`.

Important variables:
- DATABASE_URL
- JWT_SECRET
- CORS_ORIGINS
- VECTOR_DIMENSION
- CHROMA_PERSIST_DIRECTORY
- CHROMA_DOCUMENTS_COLLECTION
- CHROMA_TOOLS_COLLECTION
- LLM_SERVICE_BASE_URL
- LLM_SERVICE_GENERATE_PATH
- LLM_SERVICE_MODELS_PATH
- LLM_SERVICE_API_KEY
- LLM_SERVICE_TIMEOUT_SECONDS
- LLM_DEFAULT_GENERATION_MODEL_TYPE
- LLM_DEFAULT_GENERATION_MODEL
- LLM_DEFAULT_JUDGE_MODEL_TYPE
- LLM_DEFAULT_JUDGE_MODEL
- LLM_JUDGE_ENABLED
- LLM_DEFAULT_TEMPERATURE
- LLM_DEFAULT_MAX_TOKENS

Default local values already target SQLite + Chroma.

## LLM Service Dependency

`cents-backend` now delegates generation calls to the sibling `cents-llm` service.

Recommended local run order:

1. Start Ollama and pull a starter model (for example `qwen2.5:3b-instruct`).
2. Start `cents-llm` on `http://127.0.0.1:8100`.
3. Start `cents-backend`.

If `LLM_SERVICE_API_KEY` is configured in `cents-llm`, set the same value in backend `.env`.

## Multi-model Routing

The backend can use different models per conversation and per graph node.

### Get available models

- `GET /chat/models`
- Requires auth
- Returns model list from `cents-llm` plus backend defaults and supported node keys.

### Set conversation model config

- `GET /conversations/{conversation_id}/model-config`
- `PUT /conversations/{conversation_id}/model-config`

Payload shape for `PUT`:

```json
{
    "node_llm_configs": {
        "generation": {
            "model_type": "text-generation",
            "model": "qwen2.5:3b-instruct"
        },
        "judge": {
            "model_type": "reasoning"
        }
    }
}
```

### Per-request model override

`POST /chat` accepts optional fields:

- `node_llm_configs`: per-node map where each entry includes required `model_type` and optional `model`

Request body example:

```json
{
    "conversation_id": "...",
    "message": "Summarize the latest updates",
    "node_llm_configs": {
        "generation": {
            "model_type": "text-generation"
        },
        "judge": {
            "model_type": "reasoning",
            "model": "llama3.2:3b-instruct"
        }
    }
}
```

Rules:

- For every node that triggers an LLM call, `model_type` is required.
- `model` is optional per node. If omitted, `cents-llm` resolves the default model for that `model_type` folder.

## Notes

- Current vector embeddings are placeholders (zero vectors) until model integration is added.
- SQLite stores the relational records; Chroma stores the vectors and similarity index.
- On startup, if the local SQLite database file does not exist, it is created automatically.
- Local runtime artifacts such as `cents.db`, `.chroma/`, and `*.log` are git-ignored.
