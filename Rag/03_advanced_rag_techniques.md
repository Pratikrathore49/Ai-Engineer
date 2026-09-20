# RAG Engineer Learning Guide — Part 2: Advanced RAG Techniques (Deep Dives)

> **Trainer's Note:** Part 2 dono sections connect hote hain. `02_retrieval_improvements.md` mein concepts ki overview thi. Ye file har technique ka **pura deep-dive** hai — ek hi uniform format mein (Jenkins kahnte hain: "same problem—same six questions"):

> 1. Kya problem solve karti hai? 2. Naive RAG kyun struggle karta hai? 3. Internally kaise kaam karti hai? 4. Architecture/flow kya? 5. Kab use karein? 6. Advantages + limitations?

> Last mein: **combined advanced pipeline**, **20 interview Q&A**, **exercises**, **mistakes**, aur **decision framework** — meaning "kab kaunsa technique chuna hai ye soch nahi, ek decision tree hai."

---

## Table of Contents

1. [HyDE — Hypothetical Document Embeddings](#1-hyde--hypothetical-document-embeddings)
2. [RAG-Fusion](#2-rag-fusion)
3. [Self-RAG](#3-self-rag)
4. [CRAG — Corrective RAG](#4-crag--corrective-rag)
5. [Multi-Query RAG](#5-multi-query-rag)
6. [Parent-Child Retrieval](#6-parent-child-retrieval)
7. [Sentence-Window Retrieval](#7-sentence-window-retrieval)
8. [Contextual Compression](#8-contextual-compression)
9. [Reranking](#9-reranking)
10. [Hybrid Search](#10-hybrid-search)
11. [Query Rewriting](#11-query-rewriting)
12. [Metadata-Aware Retrieval](#12-metadata-aware-retrieval)
13. [Combining Into a Practical Advanced Pipeline](#13-combining-into-a-practical-advanced-pipeline)
14. [20 Advanced RAG Interview Questions](#14-20-advanced-rag-interview-questions)
15. [Practical Exercises](#15-practical-exercises)
16. [Common Implementation Mistakes](#16-common-implementation-mistakes)
17. [How to Decide Which Retrieval Technique to Use](#17-how-to-decide-which-retrieval-technique-to-use)

---

# 1. HyDE — Hypothetical Document Embeddings

### What problem does it solve?
Query **underspecified / too short** ho raha hai — embedded query search se miss ho rahi hai. HyDE generates a **hypothetical answer document** from the query, then embeds *that* instead of the raw query.

### Why naive RAG struggles?
Raw queries are typically 1-3 sentences, poor context. Embedding a bare query vs the full document distribution → domain mismatch (query-space vs document-space). Search in query space is weaker than direct doc-attested space.

### How it works internally
```
query  →  LLM: "Write a hypothetical passage that would answer this question."
       →  pseudo-document (fake but grounded in query)
       →  embed pseudo-document → search chunks
```
Lag: A fake document, but one *resembling* how real documents address the topic. So similarity measured in "document-space," not "query-space."

### Architecture / flow
```
User Query
   │ LLM (HyDE prompt, temperature≈0.7)
   ▼
Hypothetical document
   │ embed
   ▼
Vector search → top-k
```

### When to use
- Short/underspecified questions, especially from lay users.
- New domains where query vocabulary differs a lot from corpus vocabulary.

### Advantages
- Improves recall; simplistic code (one LLM call+embed).
- Helps when user queries lack domain terminology.

### Limitations
- Adds LLM cost/latency.
- Hypothetical doc can contain **hallucinated entities** → embeddings direct search to wrong region.
- Doesn't help keyword-heavy queries; genre mismatch possible.

### Practical implementation
```python
def hyde_embed(query: str, embed_fn, llm_fn) -> list[float]:
    pseudo = llm_fn(
        f"Write a concise factual passage that would answer this question. "
        f"Do not add opinions.\nQuestion: {query}\nPassage:",
        temperature=0.7, max_tokens=150)
    return embed_fn(pseudo)          # embed pseudo-doc, not the query

# then: result = store.chroma.query(query_embeddings=[hyde_embed(...)])
```

### Example
Query: "Bata do mujhe leave kaise milti hai?" → HyDE: "Employees earn 24 days of earned leave annually, credited at the start of each calendar year..." — fake doc embedded → matches real policy chunks.

### Interview point
**Q: HyDE vs naive query embedding?**
**A:** HyDE embeds an LLM-written hypothetical answer instead of the question — moves similarity from query-space to document-space, boosting recall on vague queries at the cost of an extra LLM call + hallucination risk in the pseudo-document.

---

# 2. RAG-Fusion

### What problem does it solve?
Multi-query retrieval results ko **merge + rerank via RRF** (Reciprocal Rank Fusion) — kai perspectives ke results ko ek unified relevancy ranking mein convert karta hai.

### Why naive RAG struggles?
Single ranked list = single perspective. When query has sub-topics ("Tell me about remote work policy AND leave AND sick pay") naive 1 cosine ranking kis dahine ki dominant single idea — chhote relevant details miss.

### How it works internally
```
1. LLM generates N sub-queries covering different facets
2. Each query → retrieve top-k
3. RRF fusion: rank all candidates by Σ 1/(60 + rank)
4. Top fused set → context
```

### Architecture / flow
```
Query
 ├─► subquery1 ──► retrieval ─┐
 ├─► subquery2 ──► retrieval ─┤─► RRF ─► top-k ─► context ─► LLM
 ├─► ...                      │
 └─► subqueryN ──► retrieval ─┘
```

### Advantages
- Recall boost + de-duplication face merged list.
- No score calibration needed (RRF uses ranks).

### Limitations
- More retrieval calls + LLM calls.
- Random sub-queries can dilute; rank-only ignores similarity magnitudes.

### Implementation — RRF fusion
```python
K = 60
def rrf_fuse(lists: list[list[str]]) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (K + rank)
    return sorted(scores, key=scores.get, reverse=True)
```
Then `context_ids = rrf_fuse(subquery_results)[:top_k]`.

### Interview point
**Q: RAG-Fusion kya? RRF usmein kaun-sa role?**
**A:** RAG-Fusion = multi-query + RRF: LLM generates facet sub-queries, each retrieves, RRF fuses rank lists (Σ1/(K+rank)) — robust order fusion handling scale differences across retrievers.

---

# 3. Self-RAG

### What problem does it solve?
Two problems:
1. **When to retrieve?** — not every prompt needs retrieval (simple FAQ, personal reply).
2. **Is the generation well-grounded?** — self-reflect both retrieval AND generation quality.

### Why naive RAG struggles?
Naive RAG **always retrieves** (even for "How are you?"), and **never self-checks** — retrieved noise + hallucinated claims flow straight through to output.

### How it works internally (reflection tokens)
Model trained/ prompted to emit **reflection tokens** controlling ground decisions:
- `Retrieve` / `NoRetrieve` — to retrieve.
- `IsRel` — retrieved chunks relevant?
- `IsSup` — generated statement supported by chunk?
- `IsUse` — statement useful?

Flow: query → decide Retrieve/NoRetrieve → if retrieve, filter by IsRel → generate with attribute tokens → **rank multiple generations** by groundedness.

### Architecture / flow
```
Query
 ├─► [Retrieve?] ─No──► generate directly
 └─► Yes → retrieve
         → filter by relevance (IsRel)
         → generate with support-reflecting tokens
         → select most grounded output (IsSup, IsUse)
```

### Advantages
- Groundedness-above hallucination; efficient (retrieval only when needed).
- Interpretable self-check.

### Limitations
- Requires a generator trained for reflection (or strong prompt), inference more expensive (multi-pass).
- Reflection quality varies; over-filtering can drop good answers.

### Practical implementation — prompt-only approximation
```python
def self_rag_generate(query: str, retrieve_fn, llm_fn) -> str:
    # Step 1: reflect — should we retrieve?
    need = llm_fn(f"Should this require document retrieval? Answer YES/NO only.\n{query}",
                  temperature=0)
    if need.strip() != "YES":
        return llm_fn(f"Reply naturally:\n{query}")
    # Step 2: retrieve
    chunks = retrieve_fn(query)
    # Step 3: reflect — are chunks relevant?
    rel = llm_fn(f"Rate relevance of each chunk to '{query}' as 0 or 1.\n"
                 + "\n".join(f"[{i}] {c[:100]}" for i, c in enumerate(chunks)))
    keep = [c for i, c in enumerate(chunks) if str(i) in rel and "0" in rel.split("\n")...]
    # Step 4: generate with support
    ...
```

### Interview point
**Q: Self-RAG vs naive RAG core difference?**
**A:** Self-RAG conditions generation on *reflection tokens* — decide whether to retrieve, whether chunks are relevant, and whether the answer statement is supported — thereby hallucination-reduction and retrieval selectivity, at cost of more inference passes.

---

# 4. CRAG — Corrective RAG

### What problem does it solve?
**Unreliable retrieval results.** Naive RAG trusts whatever top-k returns. CRAG uses a lightweight **retrieval evaluator** to grade retrieved documents → **correct the retrieval action** (perform, correct, or skip).

### Why naive RAG struggles?
No notion of "retrieval was bad." When chunks are irrelevant, naive still feeds them → misleading context → hallucination.

### How it works internally (CRAG three actions)
1. **Evaluate** retrieved docs with a score (lightweight evaluator, e.g., T5 / LLM ranking):
   - `Correct` (high) → keep retrieved, use.
   - `Ambiguous` (medium) → use web search to **augment** chunk source (knowledge-refine step).
   - `Incorrect` (low) → discard chunk, switch to web search entirely.

### Architecture / flow
```
Query
 ├─► retrieve docs
 ├─► evaluate retrieved docs → score
 │        ├─ CORRECT ─► context ──────────────┐
 │        ├─ AMBIGUOUS ─► refine+web ─► add ──┤─► Generator
 │        └─ INCORRECT ─► web search only ────┘
```

### Advantages
- Fixes catastrophic retrieval failure; adds external fallback.
- Decomposes trust: score before generating.

### Limitations
- Evaluator quality bottleneck.
- Web search dependency (access, cost, latency).
- More moving pieces to tune.

### Practical implementation — evaluator as LLM score
```python
def evaluate_docs(query: str, docs: list[str], llm_fn) -> list[float]:
    prompt = ("Given the query, score each document 0 (irrelevant) to 1 (fully relevant)."
              f"\nQuery: {query}\nReturn a JSON dict of doc_index:score.")
    out = llm_fn(prompt, temperature=0)
    return parse_json_scores(out)

def crag_flow(query: str, docs, llm_fn):
    scores = evaluate_docs(query, docs, llm_fn)
    mean = sum(scores) / len(scores)
    if mean > 0.7:
        return build_context(query, docs, llm_fn)          # CORRECT path
    if mean > 0.4:
        extra = web_search(query)                          # AMBIGUOUS -> augment
        return build_context(query, docs + extra, llm_fn)
    return build_context(query, web_search(query), llm_fn) # INCORRECT -> web only
```

### Interview point
**Q: CRAG kya correct karta hai?**
**A:** CRAG adds retrieval evaluation + action: grades docs, then Correct (use), Ambiguous (web-augment+refine), or Incorrect (drop + web) — reducing impact of bad retrieval on generation.

---

# 5. Multi-Query RAG

### What problem does it solve?
Recall misses due to **single query representation.**

### Why naive RAG struggles?
A vague/partial query ("That project timeline") produces ONE vector; variations not captured. Multi-query retrieves from several phrasings (paraphrase, focus areas) → covers more.

### How it works internally
LLM generates N query variants → each embeds & retrieves independent lists → merge (interleave / diversity / RRF).

### Architecture / flow
```
Query
 ├─► variant1 ─► embed ─► retrieve:N
 ├─► variant2 ─► embed ─► retrieve:N
 └─► variantN ─► embed ─► retrieve:N
        └──── dedupe & merge ──► top context ──► LLM
```

### When to use
High recall budget, ambiguous questions, Q&A datasets where terms surface indirectly.

### Advantages / Limitations
- Simple, effective recall bump.
- Multi LLM calls, dedupe logic, potential noise if variants poor.

### Practical implementation
```python
variants = generate_variants(query)             # LLM paraphrases (see 02)
merged = merge_ranked(
    [store.search(q, k) for q in variants],     # each returns ranked ids
    method="rrf")                                # or interleave by round-robin
```

### Interview point
**Q: Multi-query RAG vs simple query rewrite?**
**A:** Rewriting changes one query; multi-query runs *several variants* and merges — more comprehensive recall, each interpretation can surface its own relevant chunks.

---

# 6. Parent-Child Retrieval

### What problem does it solve?
Precision vs context tension — small chunks embed precisely but lack surrounding detail.

### Why naive RAG struggles?
Naive uses *one* chunk size. Small → context-thin answers; large → diluted similarity. Parent-child gives both at the same time.

### How it works internally
- **Index:** small child chunks embedded (retrieval side).
- **Return:** matching child → load its parent (paragraph/section) → give full parent to LLM.

### Architecture / flow
```
Corpus
 ├─ chunk into children (200 tok) + parents (section)
Retrieval: query ─► match child ─► map to parent ─► context = parent(s)
```

### When to use
- Long documents with sections (policies, manuals, papers, legal docs).
- Where precise retrieval and narrative context both matter.

### Advantages / Limitations
- Best-of-both precision+context; metadata links.
- Parent size can inflate token budget; needs doc structure for parent definition.

### Implementation
See `02_retrieval_improvements.md` section 11 — store children with `parent_id`, retrieve by children, fetch parents.

### Interview point
**Q: Actually retrieve child par parent dena — redundant chunks nahi?**
**A:** Index only children; parent fetched on demand by id (not indexed) — avoids redundancy, stores children compact, parent provides context only when matched.

---

# 7. Sentence-Window Retrieval

### What problem does it solve?
Retrieval at the **sentence granularity** + context expansion window around the matched sentence.

### Why naive RAG struggles?
Whole-paragraph chunks bundle irrelevant sentences → similarity diluted. A key sentence deep inside a big chunk may not surface at all.

### How it works internally
- Split into sentences; embed each sentence (or mini-window).
- On match: return sentence + N sentences around it as context block.

### Architecture / flow
```
Sentence index ──► query vector ──► best sentence(s)
    └─► expand window [s_-3 .. s_+3] ──► context to LLM
```

### Advantages / Limitations
- Maximal precision per retrieval unit; cheap.
- Sentence splitting over tables/code; window edge effects.

### Interview point
**Q: When sentence-window over parent-child?**
**A:** Prose/chat data → sentence-window ideal; structured sections/tables (policy clauses) → parent-child carries structure. Co-usable: window when prose, parent when structure.

---

# 8. Contextual Compression

### What problem does it solve?
Post-retrieval **noise cleaning** within token budget.

### Why naive RAG struggles?
top-k chunks include tangentially relevant material; LLM gets distractors.

### How it works internally
Retrieved docs → compressor (extractor LLM/classifier) → keep only query-relevant sentences → pass compressed context.

### Architecture / flow
```
retrieved chunks ─► compressor (query, chunk) ─► relevant sentences only
                                              └► combined → prompt
```

### Advantages / Limitations
- Token economy + less confusion.
- Extra model calls; extraction may drop nuance.

### Interview point
**Q: Compressor filter-empty approach kis liye ijaad?**
**A:** Compression doubles as an *irrelevance filter*: empty compress result = drop chunk → context cleaner. It behaves like probabilistic reranking with explanation support.

---

# 9. Reranking

### What problem does it solve?
Ordering from vector similarity is weak — **reorder by true relevance.**

### Why naive RAG struggles?
Embedding similarity = global direction-match; the truly answering passage may rank 12th while 4 items that "look about topic" rank higher.

### How it works internally
Two stages:
```
Stage 1: cheap broad retrieval → candidates (top-50, might be any retrieve method)
Stage 2: reform model: query+passage → per-pair score → sort ✓ top_k only
```

### When to use
- High-quality threshold (answer quality), query phrasing varies wildly.
- When recall is already decent but precision needs fixing.

### Advantages / Limitations
- Significant precision improvement; handles semantics-aware.
- Extra latency/compute; reranker-bottleneck for big candidate sets.

### Implementation
See `02_retrieval_improvements.md` section 9 — CrossEncoder reranker on `(query, passage)` pairs.

### Interview point
**Q: Rerank model quality lies where?**
**A:** Quality depends on the candidate set — reranking can't invent passages the first stage missed. Hence "retrieve broad, rerank sharp"; always keep first-stage recall adequate.

---

# 10. Hybrid Search

### What problem does it solve?
Semantic-only misses exact/entity/keyword → combine lexical + semantic.

### Why naive RAG struggles?
Embedding models are poor at rare tokens, product codes, exact names, negations, neologisms — all of which BM25 catches via literal term overlap.

### How it works internally
```
indexed: chunk vectors (embedding) AND token-Inverted (BM25)
query:   vector score (cosine) + BM25 score (lexical)
fusion:  RRF combine the two ranked lists → final top-k
```

### Advantages / Limitations
- Robust recall across query styles; best default architecture.
- Two indexes & storage; tuning α/BM25 params; synonym gap in lexical side.

### Implementation
See `02_retrieval_improvements.md` section 8 — `bm25s` + vector store + RRF.

### Interview point
**Q: Hybrid search typical boost scenario — example?**
**A:** Query "Company code HR-2045 policy" — semantic vector may not distinguish the code; BM25 exact match **pinpoints** chunk containing "HR-2045". Hybrid rescues exact-token searches.

---

# 11. Query Rewriting

### What problem does it solve?
Raw user queries are **conversational, terse, ambiguous** — improve them before embedding.

### Why naive RAG struggles?
"iska kya hoga?" — pronoun resolution, missing context, typos → garbage embeddings.

### How it works internally
LLM/rewriter: input raw query (+ history/context) → **search-oriented standalone query** → embed. May include keyword-ification, decomp, translation.

### When to use
- Conversational assistants with multi-turn history.
- Noisy query sources (chat, search bars).

### Advantages / Limitations
- Huge precision win for conversational interfaces; cheap.
- Rewritter propagates its errors; adds call lag.

### Implementation
```python
def rewrite(query, history):
    return llm(
      messages=[system_rewrite_instructions] + history[-4:] + [
        {"role":"user","content":query}])
```

### Interview point
**Q: Rewrite vs expansion?**
**A:** Rewrite = change form (resolve/repair to standalone search query); expansion = widen vocabulary (synonyms). Together they form a transformation layer in query pipeline.

---

# 12. Metadata-Aware Retrieval

### What problem does it solve?
Context window quality suffers when **filterable attributes** aren't used — search ignores date, department, doc_type.

### Why naive RAG struggles?
Embedding similarity ignores structured fields. "2026 travel policy" words may exist in a 2023 doc wording clash. Without filter, all docs compete purely on embedding.

### How it works internally
Build metadata filters from the query (LLM-extracted attributes) → apply `where` clauses → then vector search inside filtered set. Option for attribute-routing: query-language classification → per-collection.

### Architecture / flow
```
Query ─► Attribute Extractor (LLM) ─► {dept, year, region}
                              └────► chroma.query(where={"year":2026, "dept":"HR"})
```

### When to use
- Collections with metadata attributes users reference (date/region/type).
- Data catalogs, legal, finance, multi-tenant setups (tenant_id filter!).

### Advantages / Limitations
- Precision improves drastically, safer multi-tenant isolation.
- Depends on correct metadata quality at ingest; missing attribute → empty.

### Implementation
```python
def extract_filters(query: str) -> dict:
    out = llm(f"Extract filter conditions (year/dept/region) from: {query} "
              f"Return JSON." , temperature=0)
    return json.loads(out)

res = collection.query(
    query_embeddings=[embed(query)],
    n_results=5,
    where={"$and": [{"year": {"$gte": year}}, {"department": dept}]}   # dynamic
)
```

### Interview point
**Q: RAG mein tenant isolation kaise karte?**
**A:** Store `tenant_id` in every chunk's metadata; require `where={tenant_id: user.tenant}` on every query — vector DBs enforce this route, preventing cross-tenant leakage even if embeddings match.

---

# 13. Combining Into a Practical Advanced Pipeline

Sar techniques ko judna — ye ordering ka logic relevant hai:

```
User Query
   │
   ├─► [1] Query Pipeline
   │     • multi-turn history → Query Rewriting (resolution, standalone)
   │     • attribute extraction → Metadata-Aware filtering
   │     • sub-queries → Multi-Query / variant generation
   │
   ▼
[2] Retrieval (broad, cheap)
   • Hybrid Search: BM25 (exact) + Vector (semantic) — RRF fused
   • Optional: HyDE embedding instead of direct query for vague queries
   • Apply metadata filters (tenant, date, dept)
   │
   ▼
[3] Post-Retrieval
   • Rerank with Cross-Encoder → crisp top_k
   • Optional MMR for diversity (when redundancy heavy)
   │
   ▼
[4] Context Assembly + Compression
   • Parent-Child / Sentence-Window to restore context
   • Contextual Compression trims noise, enforces budget
   │
   ▼
[5] Generation
   • Prompt: "only context, cite [Source N], say don't-know"
   • temperature low
   │
   ▼
[6] Self-Correct / CRAG-style guard (when applicable)
   • evaluate retrieved relevance & answer groundedness
   • if garbage → fallback (web / re-query / respond don't-know)
```

### Important: pipeline design principle
1. **Cheap first, expensive later** — broad retrieval cheap, rerank/compress expensive, generate most expensive.
2. **Don't stack everything always** — each step adds latency/cost. Turn features on/off per query type (config flags).
3. **Each stage improves one metric** — rewriting→recall, hybrid→recall, rerank→precision, compression→focus, self-correct→trust. Track which metric each serves so tuning is targeted.

---

# 14. 20 Advanced RAG Interview Questions

**1. Naive RAG ke fundamental weak points kya hain?**
A: Single-vector similarity (keyword/entity/negation misses), no query interpretation, no multi-signal fusion, no self-check, redundant/noisy context, no metadata-awareness. Part 2 techniques target exact each.

**2. Retrieve-then-rerank kya hai?**
A: Two-stage: broad cheap retrieval (top 50) + expensive query-aware cross-encoder/LLM rerank to few best — separates recall from precision.

**3. BM25 vs embedding search?**
A: BM25: exact token overlap, IDF-weight, fast, keyword-strong. Embedding: paraphrases/semantics, weak on rare tokens/entities. Hybrid fuses both via RRF.

**4. RRF kya hai, kyun K=60?**
A: Reciprocal Rank Fusion sums 1/(K+rank) across lists; K=60 standard robust default to scale differences, no score normalization, better than raw score mixing.

**5. Cross-encoder vs bi-encoder?**
A: Bi = separate embeddings, indexable, cheap, moderate precision; Cross = query+passage jointly, high precision, per-pair cost → use cross for reranking on small candidates.

**6. HyDE ka underlying insight?**
A: Embed a hypothetical answer, not the query, to move similarity into document space — improves vague-query recall, risks pseudo-doc hallucination drifting the search.

**7. Parent-child retrieval design?**
A: Small child chunks embedded & matched; on match, fetch parent chunks (paragraph/section) for context — precision from children, context from parents.

**8. Contextual compression ka relationship to token budget?**
A: It filters chunks to relevant sentences before prompt — token usage decreases, focus increases, fewer distractors, at cost of compression LLM calls.

**9. MMR vs top-k?**
A: MMR selects with diversity penalty (rel − λ·redundancy), preventing overlapping similar chunks from dominating; λ controls diversity.

**10. Self-RAG reflection tokens kisliye?**
A: Control retrieval (Retrieve/NoRetrieve), filter relevance (IsRel), verify support (IsSup), utility (IsUse) — enable selective retrieval and grounded generation selection.

**11. CRAG three actions?**
A: Evaluator grades docs: Correct (keep), Ambiguous (augment web + refine), Incorrect (drop + web search) before generation.

**12. Query rewriting in conversational RAG?**
A: Resolves pronouns/context from multi-turn history into standalone search queries before embedding — huge practical win for chatbot UIs.

**13. Metadata-aware retrieval (tenant summary) one liner?**
A: Extracting filters from query (dept/year/region/tenant) and applying structured `where` clauses — adds structured signal embeddings lack and enables isolation.

**14. When NOT to use hybrid search?**
A: When corpus is exclusively prose and query style fully semantic (no entities/codes), or cost/storage constraints; also bilingual corpora might prefer semantic-dominant if BM25 noisy.

**15. Which single stage improves recall best?**
A: Roughly: hybrid search + broad top-k via rerank gives most consistent recall; query rewriting can be equal, data-dependent. Usually combination.

**16. Which single stage improves precision best?**
A: Cross-encoder reranking — it reorders by true query-passage relevance, fixing ordering from noisy embedding similarity.

**17. Latency vs quality trade in advanced RAG?**
A: Each stage adds ms-s range calls: rewriting(~200ms) → hyde(~+300ms) → retrieval(ms) → rerank(+5-30ms) → compression(+1-2 LLM calls). Choose on SLA and budget; serverless-consider.

**18. Evaluator retriever refresher — intro new stage lagaana order kya?**
A: 1) Query pipeline (rewrite/multi-query) → 2) Hybrid first-stage → 3) rerank second-stage → 4) compression → 5) grounding countermeasures. Order aligns with metrics each improves.

**19. Multi-query + RRF vs single best query — quality difference?**
A: Multi-query+RRF increases recall and robustness (particularly multi-facet queries) at cost of N extra retrieve/LLM calls; single rewrite cheaper but less recall coverage.

**20. Production RAG quality in one sentence?**
A: Retrieval granularity (chunk size/structure), query processing (rewrite/filter/hybrid), rerank correctness, context budget discipline, and grounding checks — collectively decide answer quality beyond the LLM itself.

---

# 15. Practical Exercises

1. **Hybrid retrofit** — apne Part 1 app mein BM25 add karo, RRF se fuse karo. Ek product-code/entity query lo ("SKU-XYZ"), bina & saath hybrid — result compare karo.
2. **Reranker plug** — `sentence-transformers` cross-encoder se retriever ko rerank karo. Retrieval distance vs rerank score: koi chunk high-distance lekin low rerank-score, uske impact ka note banao.
3. **Query rewrite A/B** — 10 queries lo, `raw vs rewritten` dono se retrieve karo, recall list compare karo (precision/recall handmade: 5 gold chunks).
4. **Multi-query + RRF** — 3 queries lo, single vs multi(4 variants)+RRF, dedupe count and top-5 quality compare.
5. **Parent-child mini-lab** — 2 long policy docs par child(200)/parent(section) index banao; query "sick leave cap" — child retreived, answer quality vs naive whole-chunk compare.
6. **CRAG approximator** — LLM evaluator (0-1 scores) se docs grade karo; threshold ke below case ke liye web-search fallback LOG karo (dummy).
7. **Decision tree walk** — apne corpus (je model examples) ko niche ke decision framework se match karo: kaunse 2-3 techniques aapke corpus effort provide kar sakte hain, aur kyu.

**Evaluation rule:** Har technique ko apne "evil test set" (hard queries: codes, negation, multi-facet, pronoun) par benchmark karo — na ki generic queries par.

---

# 16. Common Implementation Mistakes

1. **Sab techniques ek saath stack karna** — latency+debug expensive; add one per eval iteration.
2. **Rerank after tiny first-stage** (top-3 → rerank) — rerank ko material hi nahi milta; use top-30-100.
3. **Hybrid fusion score-wise (na RRF)** — scale mismatch; always rank-fuse or normalized.
4. **Parent/child size blind** — too-large parent = token blow-up later on.
5. **HyDE for keyword-heavy** — pseudo-doc embedding hurts exact search; use hybrid.
6. **CRAG without evaluator tuning** — thresholds blindly set, false fallbacks everywhere.
7. **Metadata filters hardcoded** — query-specific filters should be LLM-extracted, not static.
8. **Compression on every chunk** — call LLM per chunk na overcook budget; batch or filter cheaply first.
9. **No eval corpus while adding improvements** — without baseline query set + metrics, "improvements" are rumors.
10. **Caching na hona for rewrite/HyDE calls** — same queries repeat in production; cache em.

---

# 17. How to Decide Which Retrieval Technique to Use

Decision tree:

```
1. Kya corpus mein exact identifiers/codes/names?
        YES ─────────────┐
   NO ─────────────┐    │
   2. Corpus prose,         Hybrid search (BM25+vector)      │
      conversational-style? ──► semantic-first, +BM25        │
        |                      consider ✓                    ▼
        ▼                      | recall ok?  ──No──► parent-child /
   Lexical enough vs                 |          sentence-window
   entity-heaviness?                 ▼
        |                     precision ok?
        ▼                     ├─No──► rerank(cross-encoder)
        ▼                     └─Yes──► done (keep simple)
3. User queries terse/
   conversational?
   ├─Yes──► Query rewriting (+hybrid)
   ├─Historical ambiguity?
   │  ├─Yes──► multi-query / RAG-Fusion
   │  └─Vague vocab gap? ──► HyDE
   └────────────────────────────
4. Retrieval noise high / 
   budget tight ──► compression + MMR(diversity)
5. Trust-critical (no hal-
   lucination) ──► CRAG/Self-RAG guard rail
6. Multi-tenant / aspects ──► metadata-aware filtering
```

And the golden rule (repeat it in interviews):

> **"Har component bas ek metric improve karta hai — pehle apni weakest metric identify karo, phir usse targeted technique lagao."** Ye "RAG ease stage analysis" hai. Naive sirf isliye fail hota hai ki har weak point visibility hi nahi.

---

> **PART 2 COMPLETE — Say `continue` when you are ready for PART 3 (Production RAG + Evaluation + Real Project).**