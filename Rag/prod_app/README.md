# Production RAG (Part 3) — FastAPI + ChromaDB + Hybrid Search + Guardrails

Part 1 + Part 2 + production hygiene. Single service, config-driven.

## Project structure

```
prod_app/
├── requirements.txt
├── .env.example            # copy to .env, apne keys daalo
├── make_sample_data.py     # 2 sample HR PDFs banata hai (bina extra deps)
├── README.md
├── smoke_test.py           # offline pipeline test (mock embeddings, no network)
├── app/
│   ├── main.py             # FastAPI routes + request orchestration
│   ├── config.py           # .env -> typed settings
│   ├── schemas.py          # request/response pydantic models (API contract)
│   ├── auth.py             # X-API-Key dependency (constant-time compare)
│   ├── logging_conf.py     # structured request-scoped logging
│   ├── cache.py            # TTLCache (query -> response)
│   ├── bm25_index.py       # pure-Python BM25 (lexical arm of hybrid search)
│   ├── vector_store.py     # ChromaDB + OpenAI-compatible embeddings
│   ├── chunking.py         # structure-aware recursive chunker
│   ├── ingest.py           # PDF parse -> clean -> chunk -> embed -> index
│   ├── retrieval.py        # hybrid (BM25 + vector, RRF) -> optional rerank
│   ├── reranking.py        # cross-encoder reranker (optional dep)
│   ├── context.py          # labeled [Source N] context assembly + budget
│   ├── generation.py       # system prompt + LLM call + faithfulness check + circuit breaker
│   └── evals/
│       ├── dataset.jsonl   # golden eval cases
│       └── run.py          # retrieval + generation metrics suite
└── data/
    ├── docs/               # uploaded/sample PDFs
    └── chroma/             # persistent vector DB
```

## Setup & run

```bash
cd Rag/prod_app
.venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env        # keys daalo (OPENAI_API_KEY, API_KEY)
python make_sample_data.py    # sample PDFs
python -m app.eval.run --dataset app/evals/dataset.jsonl   # optional pre-check
uvicorn app.main:app --reload
```

Endpoints (docs at `/docs`):
- `GET /health` — status, chunk count, indexed docs.
- `POST /ingest` — upload PDF (needs `X-API-Key`).
- `POST /reindex` — rebuild BM25 after manual changes.
- `POST /query` — ask question. Returns answer + sources + faithfulness + latency breakdown.
- `GET /cache` — cache stats.

Example cURL:
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"query": "How many sick days are allowed?"}'
```

## What production components do (and why they exist here)

| Component | File | Why |
|---|---|---|
| API key auth | `auth.py` | Cheap access control; `secrets.compare_digest` constant-time — timing attack se bachav |
| Rate limit | `main.py` `_check_rate_limit` | Finish in 60s window; LLM billing control |
| Cache | `cache.py` | Repeated queries: latency + cost down. TTL from `.env` |
| Structured logging | `logging_conf.py` | Har request par retrieve hits, latency stage-wise, faithfulness — audit + debug trail |
| Hybrid search | `retrieval.py` | BM25 (exact tokens/codes) + vector (semantics), RRF fusion — naive embedding miss fix |
| Rerank | `reranking.py` | `RERANK_ENABLED=true` + optional dep → cross-encoder cleans ordering |
| Circuit breaker | `generation.py` `insufficient_context` | Weak retrieval → "I don't know" instead of hallucination + saves an LLM call |
| Faithfulness check | `generation.py` | Post-generation: unsupported claims detected, score returned in response |
| Evals | `evals/run.py` | Regression gate: retrieval metrics (no LLM) + generation metrics (LLM) |
| Error handling | `ingest.py` | Validation (extension/size) + empty-text detection → controlled 4xx, no silent garbage |

## Key logic: hybrid search + RRF

`retrieval.py::search`:
1. Vector pass — top `RETRIEVE_CANDIDATES` from Chroma.
2. Lexical pass — BM25 rank over chunk corpus.
3. RRF fusion — score = `Σ 1/(60 + rank)` over both lists, sort desc.
4. Optional cross-encoder rerank → final `top_k`.

## Eval how-to

1. Ingest apne docs. `app/evals/dataset.jsonl` mein har query ke liye `relevant_chunk_ids` bharo (apne charset ke chunk ids).
2. Run:
```bash
python -m app.eval.run --dataset app/evals/dataset.jsonl --k 5
```
3. Output: hit_rate@k / recall@k / precision@k + faithfulness / answer_relevance.
4. Koi bhi retrieval/embedding/chunk change ke baad run karke baseline se diff dekho — ye regression gate hai.

**Note:** `smoke_test.py` runs the whole pipeline offline with fake embeddings — Chroma insert, BM25, hybrid, context, circuit breaker validate karta hai bina API key ke.

## Production gaps (deliberately out of scope for this lesson)

- JWT/user-level auth (API key is service-level), multi-tenant isolation
- Async ingestion queue/workers for large corpora
- Prometheus metrics + OpenTelemetry tracing + alerting
- Object storage for source files, doc versioning/deletion API
- LLM provider failover & circuit breakers on provider outages
- `sentence-transformers` install only if reranking `RERANK_ENABLED=true`