# RAG Engineer Learning Guide — Part 3: Production RAG Architecture

> **Trainer's Note:** Aapne basic RAG (Part 1), advanced techniques (Part 2), aur eval framework (isse pehle files) dekh liya. Ab: real production mein yeh sab kaise assemble hota hai — aur **har production component kyun exist karta hai**. Har component ka "why" interview mein puchhha jata hai — matlab component provide karta hai wo certainty, na ki "kyunki best practice" bolke.

---

## The Production Request Flow

```
User Query
   │
   ├─ [CACHE hit?] ──────────► return cached response
   ▼
[1] Query Processing        → validate, sanitize, auth, history join
[2] Query Transformation    → rewrite / expand / attribute-extract / decompose
[3] Retrieval               → hybrid (BM25 + vector) + metadata filters
[4] Reranking               → cross-encoder/LLM rerank top-k
[5] Context Construction    → dedup, order, budget, source labels
[6] LLM                     → generation (temperature low, citations)
[7] Validation/Evaluation   → faithfulness/citation/grounding check
   ▼
Response (+ sources, + confidence, + support flags)
```

Har stage ke output ko log karo (structured) — kyunki production mein aapko har request ke liye **audit trail** chahiye: kya retrieve hua, kya LLM ko diya, kya LLM ne bola.

---

## Where Each Component Fits

| Component | Role in the flow | Why it exists |
|---|---|---|
| **FastAPI** | Request/inference service + ingestion endpoints | Project structure, async, validation, OpenAPI docs |
| **Vector database (ChromaDB)** | Chunk embeddings + metadata, ANN index | Persistence + fast retrieval + filtering |
| **Embedding model** | Text → vectors (ingestion & query) | Semantic similarity math |
| **LLM** | Final answer generation (reader role) | Grounded answer from context |
| **Cache** | Frequent identical/ similar queries | Latency & cost reduction |
| **Logging** | Structured logs per request/stage | Debugging, audit, monitoring source |
| **Authentication** | API keys / JWT | Only authorized access |
| **Rate limiting** | Per-user/per-key request budgets | Abuse protection, cost control, QoS |
| **Monitoring** | Metrics, alerts, dashboards, tracing | Detect degradation before users do |
| **Evaluation** | Golden-set regression, online eval | Continuous quality guard |

---

## Q: FastAPI kyun?

- Clean, typed API definition (schemas = contract).
- Async wr, dependency injection (`Depends` for auth/rate-limit).
- OpenAPI docs autogen — integration easy.
- Background tasks (ingestion re-index) + lifespan hooks (load models/collections).

---

## Q: Which Piece When — architecture tiers

**Tier 0 — Prototype (Part 1 app)**
Single service, persistent Chroma, no auth, direct DB + LLM.

**Tier 1 — Production-ish (Part 3 app)**
Same single service but: `.env` config, structured logging, auth (API key), caching, validation, error handling, reranking optional, guardrails, eval suite.

**Tier 2 — Scale (interview answer outline)**
- Separate **ingestion workers** (queue/ETL) vs **inference service** (web).
- Vector DB as standalone (Qdrant/Pinecone) + object storage for source files.
- Embedding API/GPU service.
- LLM behind API gateway (retry, rate, eviction).
- Read replicas, observability stack (OTel/Prometheus/Grafana), alerting.

> **Interview framing:** "Tier-1 architecture ka matlab hai — ek binary jo web + ingestion + eval serve karti hai, config-driven. Jab scale ke requirements aate hain, main stages ko alag services mein split karta hoon." — isse pata chalta hai ki selected approach coordinated hai, na ki speculative.

---

## Production Components — Deep Reasons

### 1. Authentication & Authorization
**Why:** Embeddings/LLM non-free; sensitive knowledge. Protect all write/read endpoints.
- **API key** (header `X-API-Key`) system/service level.
- **JWT** when you have real users → per-user permissions.
- **Tenant isolation:** every query must have `tenant_id` filter to prevent data leakage between customers.

### 2. Rate Limiting
**Why:** LLM APIs billed per token; unbounded concurrent traffic -> cost spike + degraded QoS.
- **Per-key:** token bucket per API key/IP (e.g., 10 req/min, 1000 tokens/min).
- **Token-aware:** rough `prompt_tokens + max_tokens` budget; queue/slow down the excess rather than failing all.

### 3. Caching
**What to cache:**
- **Query → full answer** (exact/similar query, short TTL — HR data changes).
- **Query → retrieval results** (embeddings of frequent queries).
- **Embedding results** (same chunk/question text).
- **HyDE/LLM-derived query artifacts** (rewritten query, pseudo-doc).
**Trade-off:** staleness risk (docs change) vs latency/cost savings. Short TTL + invalidation on doc updates.

### 4. Logging (the source of truth for debug)
**Per-request structured log:**
```json
{
  "request_id": "...", "user/tenant": "...",
  "query": "...", "rewritten_query": "...",
  "retrieved": [{"chunk_id": "...", "source": "...", "rank": 1, "score": 0.83}],
  "context_tokens": 2140, "prompt_tokens": 2450, "completion_tokens": 120,
  "latency_ms": {"retrieve": 42, "rerank": 18, "llm": 320},
  "answer": "...", "guardrail": {"faithfulness": 0.9},
  "model": "gpt-4o-mini", "embedding_model": "text-embedding-3-small"
}
```
Why structured? Because grep + dashboards + eval pipeline all consume JSON.

### 5. Monitoring & Observability
- **Metrics (Prometheus):** request rate, latency percentiles, retrieval hit rate, empty-context rate, hallucination-detection rate, token cost per request.
- **Tracing (OpenTelemetry):** stage-level spans (retrieve, rerank, LLM, guardrail) — find which stage eats latency.
- **Alerting:** p95 latency spike, faithfulness < 0.7 sustained, empty retrieval rate up.

### 6. Error Handling
- **Retries w/ exponential backoff** on transient LLM/embeddings failures.
- **Timeouts** everywhere (LLM 30s, rerank 1s).
- **Graceful degradation:** LLM down → still return retrieval-only results ("didn't generate; these sources...").
- **Circuit breakers:** sustained failures → degrade path, no more LLM calls till recovered.

### 7. Configuration (`.env`)
Everything that varies per environment: keys, model names, thresholds, top-k, caching TTL, rate limits, paths. **Never hardcode secrets.**

### 8. Security
- Secrets via env/secret manager. No keys in code or logs.
- Validate uploads (extension, size, MIME, malware scan).
- Prompt injection guard: treat user docs as untrusted data; never let ingested content override system instructions; validation of LLM output (no system-instruction leaks).
- SSRF: if ingestion fetches URLs, restrict destinations.

---

## System Design Interview — Mental Blueprint

**Given:** "Design a RAG system for 10k HR documents, 100 concurrent users, sub-2s latency."

**Answer skeleton (5 min):**

1. **Requirements:** functional (ask-over-docs, citations, freshness) / non-functional (p99 latency, cost/token budget, tenant isolation).
2. **High-level diagram:** users → API gateway (auth/rate-limit) → inference service → (query pipeline) → vector DB + BM25 → reranker → LLM → guardrail → response. Ingestion: upload → parser workers → chunk → embed → index.
3. **Component choices & trade-offs:** Chroma=False→Qdrant for scale; embedding small-1536 for cost; LLM gpt-4o-mini; cache for hot topics; keep ingestion async/queue; TTL for freshness.
4. **Data model:** chunks + metadata (source, tenant_id, updated_at), parent-child for context, versioning.
5. **Quality & eval:** golden set, RAGAS, per-stage instrumentation.
6. **Failure modes:** retrieval miss → fallback to "don't know"; LLM timeout → degrade; stale data → TTL/index rebuild; cost spike → rate-limit + caching.
7. **Scale out:** ingestion separate queue, vector DB replicas, embed/LLM providers with failover, read caching.

> **Kahani: "retrieval miss, roolback hua, toh kya — stale cache? Purane index? Throttle?" — ye hunh interview mein hota hai.**

---

**Next:** `prod_app/` folder mein production-style implementation — yeh architecture concepts ka code. Uske baad `07_interviews_and_roadmap.md`.