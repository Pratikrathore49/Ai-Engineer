# LegalMind AI — Complete Project Plan

> **The Problem:** Lawyers and legal teams spend 40-60% of their time manually
> reading through discovery documents, contracts, depositions, and case files
> looking for specific clauses, precedents, and contradictions.
>
> **The Solution:** An AI-powered document intelligence assistant that lets legal
> professionals upload case documents and ask natural language questions —
> getting grounded, cited answers in seconds instead of hours.

---

## Project Status Legend

```
✅ Complete
🔄 In Progress
⬜ Not Started
🔴 Blocked (depends on something else)
```

---

## PHASE 1 — Foundation (Backend Infrastructure)
> Everything the application needs to run. Already largely built.

| # | Feature | Status | Notes |
|---|---|---|---|
| 1.1 | FastAPI project structure | ✅ | Domain-driven layout (documents/, rag/, chat/) |
| 1.2 | PostgreSQL + pgvector via Docker | ✅ | pgvector/pgvector:pg16 image |
| 1.3 | Redis via Docker | ✅ | Conversation history, caching, rate limiting |
| 1.4 | Docker Compose (full stack) | ✅ | postgres + redis services with volumes |
| 1.5 | SQLAlchemy ORM models | ✅ | Document, DocumentChunk, ChunkEmbedding |
| 1.6 | Auto-migration on startup | ✅ | create_all() via migrations.py |
| 1.7 | Structured logging | ✅ | logger.py with levels, module names |
| 1.8 | Consistent error responses | ✅ | exception_handlers.py |
| 1.9 | Input validation | ✅ | Pydantic models with field validators |
| 1.10 | Health check endpoint | ✅ | /health with per-dependency checks |
| 1.11 | Environment configuration | ✅ | .env with all secrets |
| 1.12 | CORS middleware | ✅ | Allows React dev server |
| 1.13 | Alembic migrations | ⬜ | Replace create_all() with proper versioned migrations |

---

## PHASE 2 — Authentication & User Management
> Every real product needs users. Documents must be isolated per user.

| # | Feature | Status | Notes |
|---|---|---|---|
| 2.1 | User registration (signup) | ⬜ | Email + password |
| 2.2 | User login | ⬜ | Returns JWT access token + refresh token |
| 2.3 | JWT authentication | ⬜ | Replace API key auth for user-facing endpoints |
| 2.4 | Password hashing | ⬜ | bcrypt — never store plain passwords |
| 2.5 | Refresh token rotation | ⬜ | Auto-renew sessions without re-login |
| 2.6 | Logout + token invalidation | ⬜ | Blacklist tokens in Redis |
| 2.7 | User profile endpoint | ⬜ | GET /users/me |
| 2.8 | Document ownership | ⬜ | owner_id FK on documents table — users only see their own docs |
| 2.9 | Admin role | ⬜ | Admins can see all documents and users |
| 2.10 | Email verification | ⬜ | Verify email before allowing uploads |
| 2.11 | Password reset flow | ⬜ | Forgot password → email link → reset |

**Key library:** `python-jose` for JWT, `passlib[bcrypt]` for hashing

---

## PHASE 3 — Document Management
> Core document handling — upload, process, organize.

| # | Feature | Status | Notes |
|---|---|---|---|
| 3.1 | PDF text extraction | ✅ | pypdf |
| 3.2 | TXT extraction | ✅ | UTF-8 with latin-1 fallback |
| 3.3 | Markdown extraction | ✅ | Plain text decode |
| 3.4 | DOCX extraction | ⬜ | python-docx — lawyers use Word constantly |
| 3.5 | File size validation (10MB) | ✅ | Returns 413 |
| 3.6 | File type validation | ✅ | Returns 400 for unsupported types |
| 3.7 | Single file upload | ✅ | POST /documents/upload |
| 3.8 | **Batch upload (5-10 files simultaneously)** | ⬜ | POST /documents/upload/batch — accepts multiple files, processes concurrently |
| 3.9 | Upload progress tracking | ⬜ | WebSocket or SSE — stream processing status per file |
| 3.10 | Background indexing pipeline | ✅ | BackgroundTasks in FastAPI |
| 3.11 | Document status polling | ✅ | GET /documents/{id} — processing/ready/failed |
| 3.12 | List all documents | ✅ | GET /documents — newest first |
| 3.13 | Document detail view | ✅ | Includes chunk count |
| 3.14 | Delete document | ⬜ | DELETE /documents/{id} — cascades to chunks and embeddings |
| 3.15 | Re-index document | ⬜ | Force re-embedding with new model |
| 3.16 | Document rename | ⬜ | PATCH /documents/{id} |
| 3.17 | Document tags | ⬜ | Add custom tags — "contract", "deposition", "exhibit" |
| 3.18 | Case/Project grouping | ⬜ | Group documents by case — Case table with FK to documents |
| 3.19 | Scanned PDF detection | ⬜ | Warn user when PDF has no extractable text (image-only) |
| 3.20 | Page number extraction | ⬜ | Store page_number on each chunk — critical for legal citations |

---

## PHASE 4 — Chunking & Embedding Pipeline
> How documents are prepared for search.

| # | Feature | Status | Notes |
|---|---|---|---|
| 4.1 | Word-based chunking | ✅ | 500 words, 50-word overlap |
| 4.2 | Configurable chunk size | ⬜ | Per-document or global setting in .env |
| 4.3 | Local embeddings (sentence-transformers) | ✅ | all-MiniLM-L6-v2, 384 dims |
| 4.4 | OpenAI embeddings | ✅ | text-embedding-3-small, 1536 dims |
| 4.5 | Provider switching via .env | ✅ | EMBEDDING_PROVIDER=local/openai |
| 4.6 | Batch embedding (all chunks in one call) | ✅ | Avoids N API calls |
| 4.7 | Token count per chunk (accurate) | ⬜ | Use tiktoken instead of word count |
| 4.8 | HNSW index on embeddings | ⬜ | Required for performance above 100K chunks |
| 4.9 | Async task queue for indexing | ⬜ | ARQ or Celery — replaces BackgroundTasks for reliability |
| 4.10 | Indexing retry on failure | ⬜ | Auto-retry failed documents up to 3 times |

---

## PHASE 5 — Search & Retrieval
> Finding the right content when a user asks a question.

| # | Feature | Status | Notes |
|---|---|---|---|
| 5.1 | Semantic search (pgvector cosine similarity) | ✅ | <=> operator |
| 5.2 | Similarity threshold filtering | ✅ | Default 0.3 |
| 5.3 | Filter by document_id | ✅ | Search one doc or all docs |
| 5.4 | Configurable top_k | ✅ | 1-20 results |
| 5.5 | **Hybrid search (semantic + keyword)** | ⬜ | Combine pgvector with Postgres full-text search — critical for legal exact terms |
| 5.6 | **Reranker** | ⬜ | cross-encoder/ms-marco-MiniLM-L-6-v2 — second stage ranking |
| 5.7 | Search across specific case/project | ⬜ | Filter by case_id |
| 5.8 | Search history | ⬜ | Store past queries per user |
| 5.9 | "Find all mentions" feature | ⬜ | Return every chunk containing a term — not just top K |
| 5.10 | Metadata filtering | ⬜ | Filter by file type, upload date, tags |

---

## PHASE 6 — RAG Pipeline
> Turning retrieved content into answers.

| # | Feature | Status | Notes |
|---|---|---|---|
| 6.1 | Basic RAG pipeline | ✅ | retrieve → prompt → generate → return |
| 6.2 | Grounded system prompt | ✅ | Chunks injected with source labels |
| 6.3 | No-result fast path | ✅ | No LLM call when no chunks found |
| 6.4 | Source citations in response | ✅ | filename + chunk_index + score |
| 6.5 | Conversation history (multi-turn) | ✅ | Redis-backed, sliding window 10 messages |
| 6.6 | Page number citations | ⬜ | Show exact page number — critical for legal use |
| 6.7 | **Document comparison** | ⬜ | "Find contradictions between Contract A and Contract B" |
| 6.8 | **Clause extraction** | ⬜ | "Find all indemnification clauses across all uploaded contracts" |
| 6.9 | Streaming RAG responses | ⬜ | Stream tokens back — better UX for long answers |
| 6.10 | Answer confidence score | ⬜ | Signal to user how confident the retrieval was |
| 6.11 | Follow-up question suggestions | ⬜ | LLM suggests 3 follow-up questions after each answer |
| 6.12 | Redis failure resilience | ✅ | History save failure doesn't crash the answer |
| 6.13 | Semantic caching | ⬜ | Cache answers for semantically similar questions |
| 6.14 | Context window management (accurate) | ⬜ | tiktoken-based token counting before LLM call |

---

## PHASE 7 — React Frontend
> The user interface.

| # | Feature | Status | Notes |
|---|---|---|---|
| 7.1 | 3-column layout (sidebar + main + chat) | ✅ | Tailwind CSS |
| 7.2 | React Router navigation | ✅ | Dashboard, Documents, Search pages |
| 7.3 | TypeScript types for all API shapes | ✅ | types/index.ts |
| 7.4 | Axios API client with auth header | ✅ | lib/client.ts |
| 7.5 | Document upload with progress | ✅ | useMutation + hidden file input |
| 7.6 | Document list with status badges | ✅ | processing/ready/failed |
| 7.7 | Auto-polling for processing status | ✅ | useQuery refetchInterval |
| 7.8 | Working chat panel | ✅ | Messages, optimistic UI, sources |
| 7.9 | Conversation persistence across navigation | ⬜ | Store conversationId in localStorage |
| 7.10 | **Dark / Light theme toggle** | ⬜ | Tailwind dark: classes + localStorage preference |
| 7.11 | **User login / signup pages** | ⬜ | /login and /register routes |
| 7.12 | Auth token storage | ⬜ | Store JWT in httpOnly cookie or localStorage |
| 7.13 | Protected routes | ⬜ | Redirect to /login if not authenticated |
| 7.14 | **Batch file upload UI** | ⬜ | Drag-and-drop multiple files, progress per file |
| 7.15 | Document viewer | ⬜ | Show extracted text with highlighted cited sections |
| 7.16 | Case/Project sidebar | ⬜ | Group documents by case — matches screenshot |
| 7.17 | Search page with results | ⬜ | Full semantic search UI |
| 7.18 | Document detail page | ⬜ | Metadata, chunk count, re-index button |
| 7.19 | Settings page | ⬜ | User profile, API keys, theme |
| 7.20 | Toast notifications | ⬜ | Upload success/failure, copy answer |
| 7.21 | Mobile responsive layout | ⬜ | Collapse sidebar on small screens |
| 7.22 | Loading skeletons | ⬜ | Skeleton screens while data loads |
| 7.23 | Error boundary | ⬜ | Catch React crashes gracefully |
| 7.24 | Copy answer to clipboard | ⬜ | One-click copy button on AI responses |
| 7.25 | Export conversation | ⬜ | Download chat as PDF or TXT |

---

## PHASE 8 — Legal-Specific Features
> What makes this a legal tool, not just a generic RAG demo.

| # | Feature | Status | Notes |
|---|---|---|---|
| 8.1 | DOCX support | ⬜ | python-docx |
| 8.2 | Page number in citations | ⬜ | Store page_number per chunk during PDF extraction |
| 8.3 | Case/Matter management | ⬜ | Group documents by legal case |
| 8.4 | Document type tagging | ⬜ | Contract, Deposition, Exhibit, Pleading, etc. |
| 8.5 | Clause finder | ⬜ | "Show me all limitation of liability clauses" |
| 8.6 | Contract comparison | ⬜ | Side-by-side diff of two contracts |
| 8.7 | Contradiction detection | ⬜ | "Does anything in these docs contradict each other?" |
| 8.8 | Timeline extraction | ⬜ | "Build a chronological timeline of events from these depositions" |
| 8.9 | Key entity extraction | ⬜ | People, organizations, dates, amounts mentioned in documents |
| 8.10 | Privileged document warning | ⬜ | Flag documents marked attorney-client privileged |

---

## PHASE 9 — Production Hardening
> What you need before real users touch this.

| # | Feature | Status | Notes |
|---|---|---|---|
| 9.1 | API key authentication | ✅ | Multiple keys from .env |
| 9.2 | JWT authentication | ⬜ | Replace API keys for user-facing app |
| 9.3 | Rate limiting (in-memory) | ✅ | Sliding window |
| 9.4 | Rate limiting (Redis-backed) | ✅ | Atomic INCR + EXPIRE |
| 9.5 | Response caching (Redis) | ✅ | SHA256-keyed, 1-hour TTL |
| 9.6 | HNSW vector index | ⬜ | Performance at scale |
| 9.7 | Async task queue | ⬜ | ARQ — reliable background jobs |
| 9.8 | File storage (S3/MinIO) | ⬜ | Store original files, not just extracted text |
| 9.9 | Database connection pooling | ✅ | pool_size=5, max_overflow=10 |
| 9.10 | Graceful shutdown | ⬜ | Finish in-flight requests before stopping |
| 9.11 | Request ID tracing | ⬜ | UUID per request in all log lines |
| 9.12 | Prometheus metrics | ⬜ | Request count, latency, token usage |
| 9.13 | Dockerfile for backend | ⬜ | Multi-stage build |
| 9.14 | Dockerfile for frontend | ⬜ | nginx serving built React app |
| 9.15 | Docker Compose production config | ⬜ | Full stack: backend + frontend + postgres + redis |
| 9.16 | Environment-based config | ✅ | .env with override=True |
| 9.17 | Input sanitization | ⬜ | Strip HTML, limit special characters in queries |
| 9.18 | Document encryption at rest | ⬜ | Encrypt content column — legal docs are sensitive |

---

## PHASE 10 — Testing
> You can't ship what you can't verify.

| # | Feature | Status | Notes |
|---|---|---|---|
| 10.1 | Unit tests for chunker | ⬜ | pytest — test chunk sizes, overlap, edge cases |
| 10.2 | Unit tests for processor | ⬜ | Test PDF/TXT/MD extraction |
| 10.3 | Unit tests for retriever | ⬜ | Mock pgvector, test threshold filtering |
| 10.4 | Integration tests for upload flow | ⬜ | Full upload → index → query cycle |
| 10.5 | Integration tests for RAG pipeline | ⬜ | Known Q&A pairs, verify answer quality |
| 10.6 | API tests (pytest + httpx) | ⬜ | Test all endpoints with valid + invalid inputs |
| 10.7 | RAG evaluation dataset | ⬜ | 20-30 questions with known correct answers |
| 10.8 | RAGAS evaluation pipeline | ⬜ | Automated faithfulness + relevancy scoring |
| 10.9 | Load testing | ⬜ | locust — simulate 50 concurrent users |

---

## Completion Summary

```
Phase 1  — Foundation          ████████████░░  92%  (1 item remaining: Alembic)
Phase 2  — Auth & Users        ░░░░░░░░░░░░░░   0%  (not started)
Phase 3  — Document Mgmt       ████░░░░░░░░░░  35%  (core upload done)
Phase 4  — Chunking/Embedding  ████████░░░░░░  60%  (pipeline works, needs tuning)
Phase 5  — Search              ████░░░░░░░░░░  30%  (basic semantic search done)
Phase 6  — RAG Pipeline        ████████░░░░░░  55%  (core done, missing legal features)
Phase 7  — Frontend            █████░░░░░░░░░  35%  (layout + upload + chat done)
Phase 8  — Legal Features      ░░░░░░░░░░░░░░   0%  (not started)
Phase 9  — Production          ████░░░░░░░░░░  30%  (caching + rate limiting done)
Phase 10 — Testing             ░░░░░░░░░░░░░░   0%  (not started)

Overall: ~34% complete
```

---

## MVP Definition
> The minimum set of features needed to demonstrate this in an interview
> and call it a real product.

**MVP requires:**
- [ ] 2.1-2.3 — User signup, login, JWT auth
- [ ] 2.8 — Document ownership (users see only their docs)
- [ ] 3.4 — DOCX support
- [ ] 3.8 — Batch upload (3-5 files)
- [ ] 3.18 — Case/Project grouping
- [ ] 3.20 — Page numbers in chunks
- [ ] 5.5 — Hybrid search
- [ ] 6.6 — Page number citations
- [ ] 7.10 — Dark/Light theme
- [ ] 7.11-7.13 — Login/signup UI + protected routes
- [ ] 7.14 — Batch upload UI
- [ ] 8.2 — Page citations in answers
- [ ] 8.5 — Clause finder

**Once MVP is done you can say in an interview:**
> *"LegalMind AI is a production-ready document intelligence platform for legal professionals. Users sign up, create cases, upload their documents — PDFs, Word files, transcripts — and ask natural language questions. The system retrieves the exact relevant passages, cites the page number and document, and maintains conversation context for follow-up questions."*

---

## Task Sequence (What We Build Next)

```
Task 24  → DOCX support
Task 25  → Page numbers stored in chunks
Task 26  → JWT auth (backend — signup/login/me)
Task 27  → Document ownership + case grouping
Task 28  → Hybrid search (semantic + keyword)
Task 29  → Batch file upload (backend)
Task 30  → Frontend: login/signup pages + protected routes
Task 31  → Frontend: dark/light theme toggle
Task 32  → Frontend: batch upload UI + case grouping
Task 33  → Clause finder feature
Task 34  → Streaming RAG responses
Task 35  → Dockerfiles + production compose
Task 36  → Tests (chunker, processor, RAG pipeline)
```

---

## Interview Talking Points Per Feature

| Feature | What to say in an interview |
|---|---|
| pgvector | "I used pgvector over Pinecone because all my document metadata and vectors live in the same Postgres DB — one SQL JOIN gives me everything. No extra service, no extra round trip." |
| Hybrid search | "Semantic search alone fails for legal terms — 'indemnification' and 'limitation of liability' need exact matches. I combined pgvector cosine similarity with Postgres full-text search using Reciprocal Rank Fusion." |
| Background tasks | "Upload returns in under 100ms. Embedding 200 chunks takes 10 seconds — that runs as a background task so the user isn't blocked." |
| Redis history | "Conversation history is stored in Redis Lists with 24-hour TTL. Each turn is a JSON-encoded message. The pipeline loads the last 10 turns via LRANGE and injects them into the LLM context." |
| Dual embedding providers | "I built a provider abstraction — EMBEDDING_PROVIDER=local runs sentence-transformers for free during development. EMBEDDING_PROVIDER=openai switches to text-embedding-3-small in production. One .env change, no code changes." |
| Document ownership | "Every document has an owner_id foreign key. Every query filters by the authenticated user's ID. User A cannot access User B's documents even with a valid JWT." |
| JWT auth | "Access tokens expire in 15 minutes. Refresh tokens expire in 7 days and are stored in an httpOnly cookie — not accessible to JavaScript, preventing XSS token theft." |
