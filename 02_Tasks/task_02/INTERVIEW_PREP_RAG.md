# RAG System — Interview Questions & Answers

> These are real questions asked in AI Engineer interviews.
> Every answer is grounded in the system you actually built.

---

## SECTION 1 — System Design & Architecture

---

### Q1. Walk me through the full flow when a user uploads a PDF and asks a question.

**Upload flow:**

1. `POST /documents/upload` hits `app/documents/routes.py`
2. FastAPI reads file bytes with `await file.read()`
3. File type validated — only `.pdf`, `.txt`, `.md`. Size checked — max 10MB.
4. `processor.py` extracts text using `pypdf.PdfReader` — iterates every page, joins with newlines
5. Text cleaned — null bytes removed, 3+ newlines collapsed, multiple spaces reduced
6. `Document` row inserted into PostgreSQL with `status="processing"`, extracted text in `content` column
7. HTTP response returns immediately — `document_id` and `status="processing"`
8. FastAPI `BackgroundTask` runs `indexer.index_document()` after the response is sent:
   - `chunker.chunk_text()` splits text into 500-word overlapping windows (50-word overlap)
   - `embedder.embed_texts()` sends all chunks in one batch to `sentence-transformers/all-MiniLM-L6-v2`, gets 384-dimensional vectors back
   - For each chunk: `DocumentChunk` row inserted, then `db.flush()`, then `ChunkEmbedding` row with the vector stored in pgvector's `Vector(384)` column
   - `document.status` updated to `"ready"` on success, `"failed"` on error

**Question flow:**

1. `POST /rag/ask` hits `app/rag/routes.py`
2. Question is embedded using the **same** `embed_texts()` — produces a 384-dim vector
3. `retriever.retrieve_chunks()` runs SQL against pgvector using `<=>` cosine distance operator — returns top 5 most similar chunks
4. If no chunks pass the 0.3 similarity threshold — return "no information found" immediately, no LLM call
5. `_build_system_prompt()` injects retrieved chunk text with source labels into the system prompt
6. If `conversation_id` was passed, last 10 messages loaded from Redis and injected between system prompt and question
7. `call_llm(messages)` sends to Groq — gets back a text answer
8. Answer and question saved to Redis via `add_message()`
9. Response includes `answer`, `conversation_id`, and `sources`

---

### Q2. What is pgvector and why did you use it instead of Pinecone or ChromaDB?

**What pgvector is:**
A PostgreSQL extension that adds a native `vector` data type and similarity search operators. Install with `CREATE EXTENSION vector`, define columns as `Vector(384)`, query with `<=>` (cosine distance).

**Why pgvector over alternatives:**

| Factor | pgvector | Pinecone | ChromaDB |
|---|---|---|---|
| Cost | Free (self-hosted) | Paid ($70+/month) | Free |
| Infrastructure | Already have Postgres | Separate service | Separate process |
| Transactions | ACID — vectors + metadata in same DB | No | No |
| SQL JOINs | Yes — one query across all tables | No | No |
| Production maturity | Battle-tested Postgres | Very mature | Less mature |

**Key advantage:** Documents, chunks, and embeddings all live in the same PostgreSQL database. The retrieval query JOINs `chunk_embeddings → document_chunks → documents` in one SQL statement. With a separate vector DB you'd need two round trips — query Pinecone for IDs, then query Postgres for text.

**When you'd choose Pinecone instead:**
- 10M+ vectors (pgvector needs careful HNSW indexing at scale)
- No existing Postgres infrastructure
- Need managed scaling without ops

---

### Q3. A user uploads a 200-page PDF and gets a wrong answer. What are the three most likely causes?

**Cause 1 — Retrieval failure (most common)**

The right chunk exists but the search didn't return it.

Why: Different vocabulary ("compensation" vs "salary"), threshold too high (0.3 filtered it out), or the answer spans two chunks so neither alone has enough context.

Debug:
```json
POST /documents/search
{"query": "your exact question", "top_k": 20}
```
Scan the results. If the right chunk appears at position 12 with score 0.25, your threshold is filtering it out. Lower threshold or increase `top_k`.

**Cause 2 — Chunking boundary problem**

The answer is split across two chunks. Neither half is meaningful alone.

Example: A key table spans pages 45-46. Your 500-word chunker splits it in the middle. Each half-table is useless for answering questions about it.

Debug: Find the relevant section in the original PDF. Check which chunk it landed in by querying `document_chunks` directly. If it's split, increase chunk size or add more overlap.

**Cause 3 — LLM hallucination despite correct retrieval**

Right chunk was retrieved and injected into the prompt, but the LLM answered from training data instead of the context.

Debug: Check `sources` in the response. If `chunks_used > 0` and sources look correct but answer is wrong — the LLM ignored the context. Strengthen the system prompt with explicit constraints and set `temperature=0`.

---

### Q4. How do you evaluate whether your RAG system gives good answers?

RAG evaluation has four standard metrics from the **RAGAS framework**:

**1. Faithfulness** — Does the answer only contain information from the retrieved context? Claims not supported by any chunk = hallucination. Score: 0 to 1.

**2. Answer Relevancy** — Does the answer actually address the question? A faithful answer can still be irrelevant.

**3. Context Precision** — Of all chunks retrieved, how many were actually needed? `relevant chunks used / total chunks retrieved`. Low precision = noisy retrieval.

**4. Context Recall** — Did retrieval find ALL chunks needed to answer? Requires a ground truth dataset.

**In practice:**
Build an evaluation dataset — 20-30 questions with known correct answers from known documents. Run them through the pipeline. Check:
- Is the answer factually correct?
- Do the `sources` point to the right chunks?
- What's the average similarity score of retrieved chunks?

Automate this. Run it before every significant change to the pipeline.

---

### Q5. all-MiniLM-L6-v2 gives 384 dims. text-embedding-3-small gives 1536 dims. Why ever use the smaller one?

More dimensions generally means better quality — but it's not the only factor:

| Factor | all-MiniLM-L6-v2 (384) | text-embedding-3-small (1536) |
|---|---|---|
| Cost | Free forever | $0.02 per 1M tokens |
| Latency | ~5ms (local CPU) | ~200ms (API round trip) |
| Privacy | Data never leaves your machine | Sent to OpenAI |
| Offline | Yes | Requires internet |
| Storage per chunk | 1.5KB | 6KB |
| pgvector search speed | Faster (less math) | 4x slower |

**Choose local when:** development/testing, privacy-sensitive data, budget constraints, offline environments, general English where quality difference is small.

**Choose OpenAI when:** production with technical/multilingual documents, when retrieval quality directly affects revenue.

**Interview answer:** "We used all-MiniLM-L6-v2 for development — free, local, zero cost during iteration. The architecture is provider-agnostic via `EMBEDDING_PROVIDER` in `.env`. We'd benchmark both models on our specific corpus before deciding which to ship to production."

---

## SECTION 2 — Advanced RAG Questions

---

### Q6. What is the difference between semantic search and keyword search? When does semantic search fail?

**Semantic search** embeds the query and finds chunks with similar vector representations. Understands meaning — "exception handling" finds chunks about `try/except` even without exact word match.

**Keyword search** matches exact words or stems. Fast, predictable, no ML.

**When semantic search fails:**
- Exact codes or IDs: "Find transaction TXN-2847-XK" — you want exact match, not "similar" transactions
- Very short ambiguous queries: "API" matches everything
- Domain jargon the model never saw in training
- Numbers and dates: semantic models handle numerical specificity poorly

**Production solution — Hybrid search:**
Run both semantic AND keyword search, merge with Reciprocal Rank Fusion (RRF). Most production RAG systems use hybrid search.

---

### Q7. What is chunking and why does chunk size matter?

Chunking splits document text into smaller pieces before embedding.

**Too small (< 100 words):** Chunks lose context. A sentence without its paragraph is meaningless. More chunks = more embedding API calls = higher cost.

**Too large (> 1000 words):** One chunk covers multiple topics — the vector is diluted and matches everything vaguely. When retrieved, you're sending huge blocks to the LLM, wasting context window.

**Sweet spot: 300-600 words with 10-15% overlap.**

**Overlap** ensures sentences at chunk boundaries appear in both adjacent chunks. Without overlap, a key sentence that falls exactly on a boundary is split and loses context in both chunks.

---

### Q8. What is the "lost in the middle" problem?

LLMs pay significantly less attention to information in the **middle** of their context window. They perform best with information at the **beginning** and **end**.

**Impact on RAG:** If you retrieve 5 chunks and place them in the system prompt, chunks 2, 3, and 4 are less likely to influence the answer than chunks 1 and 5.

**Mitigations:**
1. Retrieve fewer but higher-quality chunks (top 3 not top 10)
2. Put the most relevant chunk first (already done — sorted by similarity score)
3. Use a reranker to filter chunks before passing to the LLM

---

### Q9. What is a reranker and when do you need one?

Your system does **single-stage retrieval**: embed query → cosine similarity → return top K.

A reranker adds a **second stage**. After retrieving top 20 chunks with fast cosine similarity, a cross-encoder model re-scores each chunk against the query — much more accurate but slower. Then take the top 5.

**Why two stages?**
Cosine similarity computes query and chunk independently. Cross-encoders look at query + chunk **together** — far better at judging actual relevance. Running a cross-encoder on all 10,000 chunks would be too slow — that's why you first narrow to 20.

**When you need it:** Retrieval quality is low. Users getting irrelevant results. Technical documents where precision matters.

**Common reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` from sentence-transformers — free, local.

---

### Q10. How would you scale this system to 10 million documents?

**Bottleneck 1 — pgvector search speed**
With 10M chunks, brute-force cosine search is too slow.

Fix: Add HNSW index:
```sql
CREATE INDEX ON chunk_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```
HNSW trades tiny accuracy loss for 100x speed improvement.

**Bottleneck 2 — Embedding at upload**
100 simultaneous uploads = 100 background tasks hammering the embedding model.

Fix: Move to an async task queue (Celery + Redis or ARQ). Uploads go into a queue, workers process them. Scale workers horizontally.

**Bottleneck 3 — LLM latency**
Every question waits for the LLM API.

Fix: Semantic caching — embed the query, check if a similar question was asked before (similarity > 0.95), return cached answer. Extend the existing Redis cache to be semantically aware.

**Bottleneck 4 — Single PostgreSQL**
Fix: Read replicas for search, primary for writes. At very large scale, migrate vector storage to Qdrant or Weaviate while keeping document metadata in Postgres.

---

## SECTION 3 — Scenario Questions

---

**"Users say the chatbot answers with information not in their documents. How do you fix it?"**

This is hallucination:
1. Strengthen system prompt — "You are FORBIDDEN from using knowledge not in the context below"
2. Set `temperature=0` — deterministic output
3. Add verification step — second LLM call checks if the answer is supported by retrieved chunks
4. Flag `chunks_used=0` answers with a warning in the UI
5. Implement RAGAS faithfulness scoring on a sample of answers

---

**"Your RAG works in English but poorly in Hindi. Why?"**

`all-MiniLM-L6-v2` was primarily trained on English. Hindi text produces poor embeddings — similarity search fails to find the right chunks.

Fix: Use `paraphrase-multilingual-MiniLM-L12-v2` from sentence-transformers — supports 50+ languages. Or switch to OpenAI's `text-embedding-3-small` which handles most languages better. One `.env` change with our current architecture.

---

**"How would you evaluate whether upgrading the embedding model improves things?"**

1. Build evaluation dataset — 50 questions with known correct answers from known source chunks
2. Run with current model — record whether the right chunk appeared in top 5
3. Re-embed all documents with the new model (drop and recreate `chunk_embeddings`)
4. Run the same 50 questions — compare retrieval scores
5. Also measure end-to-end answer quality, not just retrieval
6. Compare cost and latency
7. Only switch if improvement justifies the cost

This is an **offline evaluation pipeline** — standard practice before shipping model changes.

---

## SECTION 4 — Quick Fire

| Question | Answer |
|---|---|
| What is RAG? | Retrieve relevant documents at query time, inject as LLM context. Grounds answers in your data, not training data. |
| RAG vs fine-tuning? | Fine-tuning bakes knowledge into weights — expensive, stale when data changes. RAG keeps knowledge external — cheap, always current. Use both in production. |
| What is an embedding? | A list of floats representing text meaning. Similar texts → similar vectors. Distances between vectors = semantic similarity. |
| What is cosine similarity? | Measures angle between vectors. 1.0 = identical meaning, 0.0 = unrelated. Used over Euclidean because it's magnitude-invariant. |
| Why do LLMs hallucinate? | Trained to predict statistically likely tokens, not truth. Confident-sounding wrong answers can be statistically likely. RAG mitigates by grounding prompts in real retrieved text. |
| What is a context window? | Max tokens an LLM processes in one call — input + output combined. Exceeding it throws an error. |
| What is chunking overlap? | Last N words of chunk K repeated at start of chunk K+1. Ensures boundary sentences appear fully in at least one chunk. |

---

## SECTION 5 — Known Limitations of Your System

Interviewers respect engineers who know their system's weaknesses. Be able to state these and explain how you'd fix each:

1. **No user isolation** — any authenticated user searches any document. Fix: `owner_id` column on `documents`, filter in all queries.
2. **No HNSW index** — search slows above ~100K chunks. Fix: `CREATE INDEX ... USING hnsw`.
3. **Single-stage retrieval** — no reranker. Fix: add `cross-encoder/ms-marco-MiniLM-L-6-v2`.
4. **Word-count token approximation** — can cause context overflow. Fix: use `tiktoken` for exact counts.
5. **No evaluation pipeline** — no automated quality measurement. Fix: RAGAS + evaluation dataset.
6. **No semantic caching** — same question phrased differently hits the LLM every time.
7. **No document deletion** — uploaded documents can't be removed.
8. **Full history loaded** — sliding window of 10 messages is a band-aid, not a real solution. Fix: conversation summarization.
