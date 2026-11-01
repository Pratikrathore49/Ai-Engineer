# GenAI Engineer Interview Prep
## Based on Your Project — Smart Reply API + RAG Pipeline

---

## SECTION 1 — LLM Fundamentals

---

**Q1. What is an LLM and how does it generate text?**

An LLM (Large Language Model) is a neural network trained on massive amounts of text to predict the next token given a sequence of tokens. It doesn't "understand" language — it learns statistical patterns. At inference time it samples from a probability distribution over its vocabulary to generate the next word, one token at a time.

- The key mechanism is the **Transformer architecture** — specifically the self-attention layer which lets the model relate every word to every other word in the context window.
- **Temperature** controls randomness: temperature=0 always picks the highest probability token (deterministic), temperature=1 samples more freely (creative).
- LLMs are **stateless** — they have no memory between API calls. You reconstruct context by sending the full conversation history on every request.

---

**Q2. What is the difference between tokens and words?**

Tokens are the units an LLM actually processes — not words. A tokenizer splits text into subword pieces. "unhappiness" might become ["un", "happiness"] — two tokens. A single space, punctuation, or common word is usually one token.

- On average, 1 token ≈ 0.75 words, or 100 tokens ≈ 75 words.
- Token count directly affects **cost** (you pay per token) and **context limits** (GPT-4o has a 128k token context window).
- In our project we track `input_tokens` and `output_tokens` on every LLM response and log them — this is standard production practice for cost monitoring.

---

**Q3. What is prompt engineering?**

Prompt engineering is the practice of crafting the input you send to an LLM to reliably get the output you want. It's part instruction writing, part psychology — you're telling a very capable but literal system exactly what role to play and how to behave.

- **System prompt** sets the persona and rules. In our project we use a system prompt that tells the LLM to be a professional assistant, use Markdown formatting, and be concise.
- **Few-shot prompting** means including examples of input/output pairs in the prompt so the model learns the pattern from context.
- It has the highest ROI of any AI engineering skill — a better prompt often outperforms a more expensive model.

---

**Q4. What is streaming and why does it matter?**

Streaming sends LLM output tokens to the client as they are generated instead of waiting for the full response. The LLM generates one token at a time — streaming just exposes that to the client immediately.

- Without streaming, a 200-word response might take 5 seconds with the user staring at a blank screen.
- With streaming, the first word appears in ~300ms and the user sees the response being written in real time — much better UX.
- In our project we built `POST /chat/stream` using FastAPI's `StreamingResponse` with Server-Sent Events (SSE) format — `data: {token}\n\n`.

---

## SECTION 2 — RAG (Retrieval Augmented Generation)

---

**Q5. What is RAG and why do companies use it instead of fine-tuning?**

RAG is a pattern where you retrieve relevant documents from a knowledge base and inject them into the LLM prompt before asking your question. The LLM answers using that retrieved context rather than just its training data.

- **Fine-tuning** bakes knowledge into model weights — expensive, slow to update, and the model can still hallucinate. A fine-tuned model on your company docs becomes stale the moment docs change.
- **RAG** keeps knowledge outside the model — in a vector database. Updating your knowledge base is as simple as uploading a new document. No retraining needed.
- RAG also reduces hallucination because the LLM is grounded in retrieved facts, not generating from memory.

---

**Q6. Walk me through the RAG pipeline you built.**

Our RAG pipeline has two phases — indexing and retrieval.

**Indexing (happens at upload time):**
1. User uploads a PDF/TXT/MD file via `POST /documents/upload`
2. We extract clean text using `pypdf` for PDFs, UTF-8 decode for text files
3. Text is chunked into 500-word overlapping windows (50-word overlap) using `chunker.py`
4. All chunks are embedded in one batch API call to OpenAI `text-embedding-3-small`
5. Chunks saved to `document_chunks` table, vectors saved to `chunk_embeddings` table with pgvector
6. Document status updated to `"ready"`

**This all runs as a FastAPI BackgroundTask** — the upload response returns immediately with `status: processing`, indexing happens asynchronously.

- The whole indexing flow is in `indexer.py` — single responsibility, no HTTP code in it.
- We use `db.flush()` before creating each `ChunkEmbedding` to satisfy the foreign key constraint without committing mid-loop.

---

**Q7. What is a vector embedding?**

An embedding is a fixed-length list of floats (a vector) that represents the semantic meaning of a piece of text. Texts with similar meaning produce vectors that are close together in the vector space — measurable with cosine similarity.

- `text-embedding-3-small` produces 1536 floats per input. A sentence like "Redis is a cache" and "Redis stores data in memory" would have very similar vectors even though they share few words.
- This is what makes semantic search possible — you embed the user's question, find the chunks with the closest vectors, and those are the most relevant chunks.
- We store these in pgvector's `Vector(1536)` column type which supports efficient similarity search with `<=>` (cosine distance) operator.

---

**Q8. What is chunking and why do you use overlap?**

Chunking splits a large document into smaller pieces because LLMs have token limits — you can't send a 100-page PDF in one prompt. Each chunk becomes one searchable unit in the vector database.

- We chunk by words (not characters) because character splitting can break words mid-way, reducing semantic quality.
- **Overlap** means the last 50 words of chunk N appear again at the start of chunk N+1. This prevents a key sentence that falls on a chunk boundary from losing its context in both chunks.
- We skip chunks shorter than 20 words — they're usually noise like page numbers or headers and pollute search results.

---

**Q9. What is pgvector and why did you choose it over Pinecone?**

pgvector is a PostgreSQL extension that adds a native `vector` data type and similarity search operators. It turns your existing PostgreSQL database into a vector store.

- **We already had PostgreSQL running** in Docker — pgvector added zero infrastructure. With Pinecone you'd need a separate service, separate API key, separate billing.
- For learning and early production, pgvector is the right choice. Pinecone makes sense when you have billions of vectors and need managed scaling.
- pgvector supports `<->` (L2 distance), `<#>` (inner product), and `<=>` (cosine distance). For embeddings, cosine distance is standard.

---

## SECTION 3 — Production Backend

---

**Q10. Why did you use Redis for caching instead of a Python dict?**

An in-memory dict lives inside the Python process — it dies on restart, isn't shared between server instances, and can't be inspected externally. Redis is a separate process that survives restarts and is shared across all server instances.

- We cache LLM responses with a 1-hour TTL using `SET key value EX 3600`. Identical context-free questions return the cached answer without an LLM call — saves cost and latency.
- Cache keys are `cache:{sha256(normalized_message)}`. We normalize (lowercase, strip whitespace) before hashing so "Hello" and " hello " hit the same cache entry.
- We only cache context-free requests (no `conversation_id`) — once a conversation has history, the same words mean something different, so caching would return wrong answers.

---

**Q11. How does your Redis rate limiter work and why is it better than a list of timestamps?**

We use Redis `INCR` + `EXPIRE`. On each request we atomically increment a counter key `rate_limit:{api_key}`. On the first request (count==1) we set a 60-second expiry. If count exceeds 20 we return 429.

- **Atomic** means Redis processes `INCR` as a single indivisible operation. With a Python list of timestamps, two simultaneous requests could both read `count=19`, both think they're under the limit, and both get through — a race condition. Redis prevents this.
- The key auto-expires after 60 seconds — the window resets without any cleanup code.
- We return a `Retry-After` header on 429 responses so clients know exactly when to retry.

---

**Q12. What is the difference between a background task and an async endpoint in FastAPI?**

An `async` endpoint runs on FastAPI's event loop — it's concurrent but still blocks the response until it completes. A `BackgroundTask` runs **after** the response is already sent to the client.

- In our upload endpoint, extracting text and saving the document row happens inline (blocking, fast). The heavy work — chunking + embedding — is queued as a `BackgroundTask`. The client gets `status: processing` in ~100ms.
- Background tasks are simple for single-instance apps. For multi-instance production you'd use a proper task queue like Celery + Redis or a cloud queue (SQS, Pub/Sub).
- The background task creates its own DB session — it cannot reuse the request's session because FastAPI closes that session when the response is sent.

---

**Q13. Why do you use connection pooling for PostgreSQL?**

Opening a new TCP connection to PostgreSQL takes 5-50ms — it does a handshake, authentication, SSL negotiation. Under load that overhead compounds. A connection pool keeps N connections open and reuses them.

- We configure `pool_size=5` (always-open connections) and `max_overflow=10` (temporary extra connections under spike load). Max total = 15 concurrent DB connections.
- `pool_pre_ping=True` sends a `SELECT 1` before handing a connection to your code. If the connection died (PostgreSQL restarted), it's discarded and a fresh one opened silently.
- SQLAlchemy manages the pool automatically — you just call `get_db()` and get a live connection.

---

**Q14. How does your conversation memory work?**

Each conversation has a UUID. Messages are stored in Redis as a List — one JSON-encoded `{"role": "user/assistant", "content": "..."}` string per list item. On each request we load the full history with `LRANGE key 0 -1` and prepend the system prompt before sending to the LLM.

- Redis Lists preserve insertion order — `RPUSH` appends to the right, `LRANGE 0 -1` retrieves in chronological order.
- Conversations have a 24-hour TTL, refreshed on every message. Active conversations never expire mid-session.
- This is stateless at the API layer — any server instance can handle any request as long as it has Redis access.

---

## SECTION 4 — Architecture & Design

---

**Q15. Why did you structure the project into domain folders (documents/, core/) instead of flat files?**

Flat structure works for 5 files. At 20+ files you can't find anything and every file imports from everywhere — a change in one file breaks five others.

- Domain folders (documents/, chat/, core/) group everything related to one feature together. When you work on documents you only look in `app/documents/`. When you work on auth you look in `app/core/`.
- This matches how real companies structure services — each domain is potentially its own microservice later.
- We migrated gradually — only moving files when we worked on them, not doing a big-bang reorganization. This is the production approach: you can't stop shipping to refactor everything at once.

---

**Q16. What is the fail-fast pattern and where did you use it?**

Fail-fast means detecting a broken dependency at startup and refusing to start rather than running in a degraded state and failing mysteriously on the first request.

- In `main.py` lifespan, we call `verify_redis_connection()` and `verify_db_connection()` before accepting any traffic. If either fails, the server exits with a clear log message.
- This is far better than starting, appearing healthy, then crashing on the first `/chat` request with a cryptic `ConnectionRefusedError`.
- The pattern is also used in `embedder.py` — if `OPENAI_API_KEY` is not set, we raise `RuntimeError` immediately rather than making an API call and getting a confusing auth error.

---

**Q17. Why do all your Redis keys have TTLs and namespace prefixes?**

Two separate concerns — correctness and organization.

- **TTL (Time To Live):** Redis stores everything in RAM. Without TTLs, keys accumulate forever and eventually the server runs out of memory. Every key in our system expires: cache keys after 1 hour, rate limit keys after 60 seconds, conversation keys after 24 hours.
- **Namespace prefixes:** `cache:`, `rate_limit:`, `conversation:` — when multiple features share one Redis database, prefixes prevent key collisions. They also make it easy to inspect keys by feature: `KEYS rate_limit:*` shows only rate limiter state.
- In production you'd use separate Redis databases (DB 0, 1, 2) or separate Redis instances per feature for stronger isolation.

---

## SECTION 5 — Questions They Will Ask You

---

**Q18. What would you improve about your RAG pipeline if you had more time?**

Several things:

- **Semantic chunking** instead of fixed word windows — split at sentence or paragraph boundaries to preserve semantic units.
- **Re-ranking** — after retrieving top-K chunks by vector similarity, use a cross-encoder to re-rank them by relevance to the actual question. This improves precision significantly.
- **Celery task queue** instead of FastAPI BackgroundTasks for indexing — proper retry logic, visibility into task status, horizontal scaling of workers.

---

**Q19. How would you evaluate if your RAG system is working well?**

Three metrics matter:

- **Retrieval precision** — are the chunks retrieved actually relevant to the question? Measured with human-labeled test sets or LLM-as-judge.
- **Answer faithfulness** — is the LLM's answer grounded in the retrieved chunks, or is it hallucinating beyond them? Tools like RAGAS measure this automatically.
- **End-to-end answer quality** — does the final answer actually help the user? This requires human evaluation or a strong LLM judge comparing generated answers to reference answers.

---

**Q20. What is hallucination and how does RAG reduce it?**

Hallucination is when an LLM generates confident-sounding text that is factually wrong. It happens because LLMs are trained to produce fluent text, not necessarily accurate text.

- RAG reduces hallucination by grounding the LLM in retrieved facts. Instead of "what do you know about X", the prompt becomes "here are the relevant documents, answer based only on these."
- You further reduce hallucination by adding instructions in the system prompt: "If the answer is not in the provided context, say you don't know."
- It doesn't eliminate hallucination entirely — the LLM can still misinterpret retrieved text — but it significantly reduces it compared to pure parametric knowledge.

---

*Based on Smart Reply API v2 — FastAPI + Redis + PostgreSQL + pgvector + OpenAI RAG pipeline*

---

## SECTION 6 — System Design & Architecture

---

**Q21. How would you design the architecture for a production AI engineer project from scratch?(!imp)**

I would split it into three layers — ingestion, serving, and storage.

- **Ingestion layer:** an async pipeline that accepts documents, extracts text, chunks it, generates embeddings via an embedding model, and stores vectors in a vector database. This runs as background workers, not inline with HTTP requests.
- **Serving layer:** a REST API (FastAPI) that takes a user query, embeds it, retrieves top-K relevant chunks from the vector DB, builds a prompt with those chunks as context, calls the LLM, and streams the response back.
- **Storage layer:** PostgreSQL for structured data (documents, chunks, metadata), Redis for caching and session state, a vector store (pgvector or Pinecone) for embeddings. Each store is chosen for what it does best — no single database handles everything.

---

**Q22. What is the difference between RAG and fine-tuning? When would you use each?(!imp)**

Fine-tuning updates the model weights using your own data — the knowledge gets baked into the model itself. RAG keeps the model weights frozen and retrieves relevant information at query time from an external database.

- Use **fine-tuning** when you want the model to learn a specific style, tone, or format — not new facts. For example, training a customer service bot to always respond in a specific brand voice.
- Use **RAG** when your knowledge base changes frequently, is large, or when you need the model to cite sources. Company documentation, legal documents, product catalogs — all RAG.
- In most real products the answer is **both** — a fine-tuned model for behavior + RAG for knowledge. But RAG alone solves 80% of use cases at a fraction of the cost of fine-tuning.

---

**Q23. What is an AI Agent and how is it different from a standard LLM call?(!imp)**

A standard LLM call takes an input and returns an output — one round trip, deterministic flow. An AI agent is a loop where the LLM decides what action to take, executes it, observes the result, and decides the next action — repeating until the task is done.

- Agents use **tools** — functions the LLM can call (search the web, run code, query a database, call an API). The LLM reasons about which tool to use and with what arguments.
- The loop is: think → act → observe → think again. This is called the ReAct pattern (Reasoning + Acting).
- We built a simple version of this in our project — the LLM can call calculator, weather, and web search tools. The `services.py` loop checks for `tool_calls` in the response, executes them, appends results, and calls the LLM again for the final answer.

---

**Q24. What is LangChain and when would you use it?**

LangChain is a Python framework that provides abstractions for building LLM applications — chains, agents, memory, retrievers, document loaders. It reduces boilerplate for common patterns.

- It's useful for **rapid prototyping** — you can build a RAG pipeline in 20 lines using LangChain's built-in components instead of 200 lines of custom code.
- The tradeoff is **abstraction overhead** — LangChain hides what's happening, which makes debugging hard. Many teams start with LangChain then migrate to custom code once they need more control.
- A good interview answer: "I understand what LangChain does under the hood because I built the same patterns from scratch — chunking, embedding, retrieval, prompt construction. LangChain automates those steps."

---

**Q25. What is cosine similarity and why is it used for vector search?(!imp)**

Cosine similarity measures the angle between two vectors, not their magnitude. Two vectors pointing in the same direction have cosine similarity of 1.0 (identical meaning), opposite directions gives -1.0, perpendicular gives 0.0.

- We use cosine similarity instead of Euclidean distance because embedding vectors can have different magnitudes but similar directions — cosine ignores magnitude and focuses purely on direction (meaning).
- For example, a short sentence and a long paragraph about the same topic might have very different vector lengths but nearly the same direction — cosine similarity correctly identifies them as semantically close.
- In pgvector the `<=>` operator computes cosine distance (1 - cosine similarity). Lower distance = more similar. `ORDER BY embedding <=> query_vector LIMIT 5` finds the 5 most relevant chunks.

---

**Q26. What is context window and why does it matter for production AI systems?**

The context window is the maximum number of tokens an LLM can process in a single call — both input and output combined. GPT-4o has 128k tokens, older models had 4k.

- Everything you send to the LLM — system prompt, conversation history, retrieved chunks, user question — must fit within the context window. If it overflows, you get an error or the model silently truncates your input.
- In RAG systems this is the main constraint. If you retrieve too many chunks they overflow the context. You need to select and rank chunks carefully to stay within limits.
- **Context management** is a core AI engineering skill — summarizing old conversation turns, limiting retrieved chunks, truncating long documents before sending to the LLM.

---

**Q27. How would you handle LLM API failures in production?**

Three layers of defense — retry, fallback, and graceful degradation.

- **Retry with exponential backoff:** LLM APIs occasionally return 429 (rate limit) or 500 errors. Retry 3 times with increasing delays (1s, 2s, 4s) before giving up. The `tenacity` library handles this cleanly in Python.
- **Fallback model:** if GPT-4o fails, try GPT-4o-mini. If the primary provider (OpenAI) is down, try a secondary provider (Groq, Anthropic). We already have Groq wired up in our project.
- **Graceful degradation:** if all retries and fallbacks fail, return a clear error to the user — "AI service is temporarily unavailable, please try again in a moment." Never let an LLM failure crash the entire application.

---

**Q28. What is the difference between synchronous and asynchronous in FastAPI?**

Synchronous (`def`) routes run in a thread pool — they block the thread until they complete. Asynchronous (`async def`) routes run on the event loop — they yield control while waiting for I/O, allowing other requests to be handled concurrently.

- Use `async def` for I/O-bound operations like HTTP calls to LLM APIs, Redis reads, database queries — anything where you're waiting for a network response.
- Use `def` for CPU-bound operations like text processing, chunking, PDF parsing — these should run in threads to avoid blocking the event loop.
- A common mistake is using `async def` for a blocking operation — it runs on the event loop and blocks every other request. If your function does CPU work, use `def` or run it with `asyncio.run_in_executor`.

---

**Q29. What is Docker and why does every production AI service use it?**

Docker packages your application and all its dependencies into a container — a self-contained unit that runs identically on any machine. The container includes the Python version, libraries, environment variables, everything.

- Without Docker: "it works on my laptop" is a common problem — different Python versions, different library versions, different OS configurations between dev and prod.
- With Docker: you define everything in a `Dockerfile`. The container runs identically in local dev, CI, staging, and production. Zero environment drift.
- For AI services specifically, Docker is essential because of complex dependency chains — pgvector, CUDA drivers, specific Python versions. Docker freezes all of that into a reproducible image.

---

**Q30. What is Docker Compose and how did you use it?**

Docker Compose is a tool to define and run multi-container applications with a single `docker-compose.yml` file. Instead of running three separate `docker run` commands and remembering all the flags, one `docker compose up -d` starts everything.

- In our project `docker-compose.yml` defines two services — `postgres` (pgvector image) and `redis` (alpine image). Both have named volumes for data persistence and healthchecks to confirm they're ready before the app starts.
- Named volumes (`postgres_data`, `redis_data`) mean data survives `docker compose down` and restarts — you don't lose your database every time you restart the stack.
- In production you'd use Kubernetes instead of Compose, but Compose is the standard tool for local development and small deployments.

---

**Q31. How would you make your AI application scalable to handle high traffic?**

Horizontal scaling with shared state — run multiple instances of the API, all talking to the same Redis and PostgreSQL.

- **Stateless API servers:** our FastAPI app stores no state in memory — conversations are in Redis, data is in PostgreSQL. So you can run 10 identical instances behind a load balancer and any instance can handle any request.
- **Separate the heavy work:** embedding and indexing are slow. Move them to dedicated worker processes (Celery workers) that pull jobs from a queue (Redis or RabbitMQ). Scale workers independently of the API.
- **Cache aggressively:** LLM calls are expensive and slow. Redis cache means identical questions never hit the LLM twice. At scale, cache hit rate is one of the most important metrics to track.

---

**Q32. What is semantic search and how is it different from keyword search?**

Keyword search (like SQL `LIKE` or Elasticsearch) matches exact words. Semantic search matches meaning — it finds documents that are about the same topic even if they use completely different words.

- Keyword search for "car maintenance" misses documents about "vehicle servicing" or "automobile repair" — different words, same meaning.
- Semantic search embeds both the query and documents into vectors. "car maintenance" and "vehicle servicing" produce similar vectors — they're close in the vector space — so semantic search finds both.
- The tradeoff: semantic search requires embedding every query at runtime (an API call), while keyword search is pure SQL. In production you often use both — keyword search for exact matches, semantic search for conceptual matches — this is called hybrid search.

---

**Q33. What is a vector database and what are the main options?**

A vector database stores high-dimensional vectors and supports efficient nearest-neighbor search — finding the K most similar vectors to a query vector. This is the core operation of semantic search.

- **pgvector** — PostgreSQL extension. Best for teams already using Postgres, moderate scale (tens of millions of vectors). Free, no extra infrastructure.
- **Pinecone** — fully managed cloud vector database. No infrastructure to manage, scales to billions of vectors. Paid, but very developer-friendly.
- **Weaviate / Qdrant / Milvus** — open-source dedicated vector databases. More control than Pinecone, more features than pgvector. Good for large-scale self-hosted deployments.

---

**Q34. What is LLM context management and how do you handle long conversations?**

As a conversation grows, the message history can exceed the model's context window. You need strategies to keep the context within limits without losing important information.

- **Sliding window:** keep only the last N messages. Simple but loses early context — the model forgets what was said at the start of the conversation.
- **Summarization:** periodically compress old messages into a summary using the LLM itself. "Summarize the conversation so far in 3 sentences." Store the summary and use it instead of the full history.
- **Relevance-based selection:** embed each message and retrieve only the most relevant past messages for the current question — treating conversation history like a mini RAG system. Most sophisticated but most expensive.

---

**Q35. What are the main LLM providers and how do they differ?**

The main providers are OpenAI, Anthropic, Google, and Meta (open source). Each has different models, pricing, strengths, and API designs.

- **OpenAI** — GPT-4o is the benchmark for quality. Best tool calling, best instruction following, most widely supported by frameworks. Most expensive.
- **Anthropic Claude** — strong at long documents, safety, and reasoning. Claude has a 200k token context window — best for processing entire books.
- **Groq** — not a model provider, but an inference provider. Runs open-source models (Llama, Mixtral) at extremely high speed (hundreds of tokens/sec) for free. We use Groq in our project for chat because it's fast and free.
- **Meta Llama** — open-source models you can run locally or self-host. Free, private, no API calls. Runs on Ollama locally.

---

**Q36. How do you secure an AI API in production?**

Multiple layers — authentication, authorization, rate limiting, input validation, and output filtering.

- **Authentication:** every request requires an API key in the `X-API-Key` header. We validate it against a set of allowed keys loaded from environment variables. In production these would be database-backed with per-user keys.
- **Rate limiting:** Redis-backed counter limits each API key to 20 requests per minute. Prevents abuse and protects LLM cost budget. Return 429 with `Retry-After` header.
- **Input validation:** Pydantic validates every request body — message length limits (max 2000 chars), UUID format for conversation IDs. Reject malformed requests before they reach the LLM.
- **Output filtering** (not yet built): in production you'd add a content moderation layer on LLM output to catch harmful content before it reaches the user.

---

**Q37. What is the ReAct pattern in AI agents?**

ReAct stands for Reasoning + Acting. It's the standard pattern for building AI agents where the LLM alternates between thinking about what to do and taking an action.

- The loop: the LLM receives a task → reasons about it (chain of thought) → decides to call a tool → executes the tool → receives the result → reasons again → decides next action or gives final answer.
- The key insight is that giving the LLM tool results to reason about produces dramatically better answers than asking it to answer from memory alone.
- In our project we implement a simple one-step version: LLM decides on a tool call → we execute it → pass result back → LLM gives final answer. Full ReAct agents do this in a loop until the task is complete.

---

**Q38. What is observability in AI systems and why is it harder than regular software?**

Observability means being able to understand what your system is doing from its outputs — logs, metrics, and traces. In AI systems it's harder because LLM behavior is non-deterministic and failures are often silent and subtle.

- **Traditional software** fails loudly — an exception crashes the process. **LLM applications** can fail silently — the model returns a response that looks valid but gives wrong information, misuses a tool, or ignores instructions.
- You need to log every LLM call: model name, input token count, output token count, latency, tool calls made. We do this in `llm.py` on every call.
- Production AI systems also need **LLM-specific metrics** — answer quality scores, hallucination rates, tool call success rates. Tools like LangSmith, Helicone, or custom dashboards provide this visibility.

---

**Q39. What is the difference between zero-shot, few-shot, and chain-of-thought prompting?**

These are three techniques for getting better outputs from LLMs by changing how you frame the prompt.

- **Zero-shot:** just give the instruction with no examples — "Classify this email as spam or not spam: ..." The model uses its training to figure out the task. Works for simple, common tasks.
- **Few-shot:** include 2-5 examples of input/output pairs before the actual input. The model learns the pattern from your examples. Much more reliable for specific formats or edge cases.
- **Chain-of-thought:** add "Let's think step by step" or show reasoning in your examples. Forces the model to reason explicitly before giving a final answer. Dramatically improves accuracy on math, logic, and multi-step problems.

---

**Q40. How would you estimate and control LLM costs in production?(!imp)**

LLM costs can spiral quickly if unchecked — a single misconfigured prompt can cost hundreds of dollars at scale.

- **Track tokens on every call:** log `input_tokens` and `output_tokens` per request. We do this in `llm.py`. Multiply by per-token pricing to get cost per call. Set alerts if cost per request exceeds a threshold.
- **Cache aggressively:** identical questions should never hit the LLM twice. Our Redis cache with 1-hour TTL means repeated questions cost nothing after the first call. At scale, cache hit rate is the single biggest lever on LLM cost.
- **Model selection by task:** use cheap models (GPT-4o-mini at $0.15/million tokens) for simple tasks, expensive models (GPT-4o at $5/million tokens) only when quality justifies it. A classification task doesn't need GPT-4o — GPT-4o-mini does it just as well for 30x less cost.

---

*Last updated — 20 general GenAI engineering questions added (Q21-Q40)*
