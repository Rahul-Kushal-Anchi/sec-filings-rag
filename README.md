# sec-rag-assistant

![CI](https://github.com/Rahul-Kushal-Anchi/sec-rag-assistant/actions/workflows/ci.yml/badge.svg)

Production-grade RAG system over SEC 10-K and 10-Q filings — hybrid retrieval, cross-encoder reranking, structured citations, and an offline evaluation harness.

## Demo

> **Q:** What was Apple's total net sales for fiscal year 2025?  
> **A:** Apple's total net sales for fiscal year 2025 were **$416,161 million ($416.2 billion)**, up 6% year-over-year. (AAPL 10-K · 2025-09-27 · confidence 0.997 · 18s)

<!-- Loom walkthrough coming soon -->

## Architecture

```
Query
  → OpenAI text-embedding-3-small (1,536-dim)
  → Dense retrieval: pgvector HNSW (cosine, top 20)
    Sparse retrieval: BM25 in-memory index (top 20)
  → Reciprocal Rank Fusion (k=60, top 20 merged)
  → Cross-encoder reranking: ms-marco-MiniLM-L-6-v2 (top 5)
  → Claude Sonnet 4.6 (Anthropic primary / GPT-4o-mini fallback)
  → Cited answer with filing ticker, date, page, and snippet
```

## Metrics (measured on live corpus)

| Metric | Value |
|---|---|
| Corpus | 81,864 chunks · 125 filings · 25 tickers |
| Embedding model | text-embedding-3-small (1,536 dims) |
| Chunk size | 800 chars · 150-char overlap |
| Confidence — financial queries | 0.997 |
| Confidence — risk factor queries | 0.862 |
| End-to-end latency | ~15–18 seconds |
| Test coverage | 78% · 57 tests passing |
| LLM | claude-sonnet-4-6 · gpt-4o-mini fallback |

## Tech Stack

**Backend:** FastAPI · SQLAlchemy 2.0 async · pgvector 0.8.2 · PostgreSQL 16 · LangSmith tracing · Prometheus metrics

**Retrieval:** pgvector HNSW index · rank-bm25 · sentence-transformers cross-encoder

**Generation:** Anthropic Claude Sonnet 4.6 · OpenAI embeddings + fallback · tenacity retry

**Frontend:** React 18 · TypeScript · Tailwind CSS · Vite

**Infrastructure:** Docker Compose · GitHub Actions CI · pytest (57 tests, 78% coverage)

## Core Components

Three files written by hand — the design decisions an interviewer will probe:

| File | What it does |
|---|---|
| `backend/app/ingest/chunker.py` | Recursive section-aware splitter: 800-char target, 150-char overlap, SEC-aware separator hierarchy |
| `backend/app/retrieval/hybrid.py` | Reciprocal Rank Fusion (k=60): avoids score-scale mismatch between cosine similarity and BM25 |
| `backend/app/generation/answerer.py` | Full pipeline orchestrator: dense+BM25 → RRF → rerank → prompt → LLM → cited answer |

## Quick Start

```bash
git clone https://github.com/Rahul-Kushal-Anchi/sec-rag-assistant
cd sec-rag-assistant
cp .env.example .env          # add OPENAI_API_KEY and ANTHROPIC_API_KEY
docker compose up -d postgres
cd backend
pip install -e ".[dev]"
alembic upgrade head
uvicorn app.main:app --reload  # API at http://localhost:8000

# frontend
cd ../frontend && npm install && npm run dev   # UI at http://localhost:5173
```

## Project Structure

```
sec-rag-assistant/
├── backend/
│   ├── app/
│   │   ├── ingest/       # HTML parser, embedder, recursive chunker
│   │   ├── retrieval/    # Dense (pgvector), BM25, hybrid RRF, reranker
│   │   ├── generation/   # LLM client, prompts, answer orchestrator
│   │   ├── eval/         # EvalRunner, scorer, 20-question golden set
│   │   └── api/          # FastAPI routes, SSE streaming
│   └── tests/            # 57 tests, 78% coverage
└── frontend/             # React chat UI with citations panel
```

## Evaluation

The eval harness (`backend/app/eval/`) runs 20 golden questions across AAPL and JPM filings, measuring keyword hit rate, average confidence, and latency. All expected keywords are derived from actual filing content.

```bash
cd backend
python -m app.eval.run
```

## License

MIT
