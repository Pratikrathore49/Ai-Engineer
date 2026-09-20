# RAG Engineer Learning Guide — Part 2: Why Naive RAG Fails + Retrieval Improvements

> **Trainer's Note:** Part 1 mein aapne naive RAG bana liya. Ab part 2 mein pehla sawaal: **"Naive RAG kyun fail hota hai?"** Jab aap ye 10 failure modes samajh jaoge, toh aapko ye samjh aayega ki har advanced technique kis *specific* failure ko fix karne ke liye bani hai. Yehi interview ki golden chains hai: "Yeh technique X problem solve karti hai" — aur aapko X ka naam sikha rahe hain.

---

## Table of Contents

- [Why Naive RAG Fails — The 10 Failure Modes](#why-naive-rag-fails--the-10-failure-modes)
- [Retrieval Improvements](#retrieval-improvements)
  - [1. Better Chunking Strategies](#1-better-chunking-strategies)
  - [2. Metadata Filtering](#2-metadata-filtering)
  - [3. Similarity Search Limitations](#3-similarity-search-limitations)
  - [4. Maximum Marginal Relevance (MMR)](#4-maximum-marginal-relevance-mmr)
  - [5. Multi-Query Retrieval](#5-multi-query-retrieval)
  - [6. Query Transformation](#6-query-transformation)
  - [7. Query Expansion](#7-query-expansion)
  - [8. Hybrid Search (BM25 + Vector)](#8-hybrid-search-bm25--vector)
  - [9. Reranking](#9-reranking)
  - [10. Cross-Encoder Rerankers](#10-cross-encoder-rerankers)
  - [11. Parent-Child Chunking](#11-parent-child-chunking)
  - [12. Sentence-Window Retrieval](#12-sentence-window-retrieval)
  - [13. Contextual Retrieval (Anthropic)](#13-contextual-retrieval-anthropic)
  - [14. Contextual Compression](#14-contextual-compression)

---

## Why Naive RAG Fails — The 10 Failure Modes

Naive RAG (simple embed-query → top-k → LLM) ka problem: **ek hi embedding comparison ka result LLM ko diya gaya.** Har failure mode niche par hey — eksilter.

### Failure modes at a glance

| # | Failure Mode | Keyab hai? | Fix (Part 2 techniques) |
|---|---|---|---|
| 1 | **Keyword/Entity miss** | Semantic embedding "SKU-2045" ya "HR-104" jaise exact strings ko miss kar deta hai | Hybrid search (BM25) |
| 2 | **Ambiguous query** | "Python policy?" — language ya snake? | Multi-query, query expansion/rewriting |
| 3 | **Negation & nuance** | "NOT eligible for X" chunk niche similarity, fosse galat | Reranking, cross-encoder |
| 4 | **Chunk boundary breaks** | Relevant info do chunks mein tooti | Overlap, parent-child, sentence-window |
| 5 | **Swarm of duplicate chunks** | Overlap/duplicate se top-k mein redundant content | MMR (deduplication) |
| 6 | **Whole-doc irrelevance** | Chunk relevant hai, but doc spirit galat | Metadata filtering, contextual retrieval |
| 7 | **Query info-hole** | Query choti hai kuch bol nahi paayi "last year's bonus scheme" | HyDE, query rewriting |
| 8 | **No domain calibration** | Embedding afwa chhal similarity, domain score type hat | Fine-tuned embedding, reranking |
| 9 | **Context budget overflow** | Top-k bado chunks → LLM nahi bache tokens | Contextual compression |
| 10 | **Multi-hop questions** | "Who managed the project named in doc 3?" | Multi-step agents (Part 3), RAG-Fusion |

### The core principle

> **Naive RAG single-source-of-truth ka issue:** sirf `cosine(query_vector, chunk_vector)` par bharosa. Par similarity ≠ relevance, aur relevance multifaceted hai (keyword match + semantic match + metadata + recency + source authority). Advanced RAG = **multiple relevance signals ko combine karna** (hybrid) + **weak signals ko refine karna** (rerank, compress).

Ye concept baar baar aayega. Part 2 ka poora content isi ek line ka expansion hai.

---

# Retrieval Improvements

> Ab har improvement technique ko detail mein dekhte hain. Har ek ka format: **Concept / Why / How / Code / When / Interview point.**

---

## 1. Better Chunking Strategies

### Concept
Part 1 ka basil chunking (recursive) — ab better band-ke strategies: **structure-aware** (chunk philosophy ke saath headings), **section-aware**, **table-aware**. Core idea: chunk = *retrievable unit of meaning*, token theke nahi.

### Why?
Basil recursive chunking structure ko ignore karta hai — headings/tables/code blocks tod dalta hai. Structure-aware chunking retrieval granularity aur answered quality dono ko improve karta hai.

### Best practices matrix
| Data type | Approach |
|---|---|
| Markdown/HTML | Split by headings (`##`, `<h2>`), headings ko chunk context mein append karro |
| Tables | Har row = chunk, column headers = chunk text mein daalo |
| Code | Split by function/class boundaries |
| Legal/tech docs | Split by clause/section, keep section IDs in metadata |
| Chat/conversations | Split per turn or per topic window |

### Practical implementation — structure-aware (Markdown)
```python
def split_by_headings(md: str, min_chars: int = 200) -> list[dict]:
    """Split markdown by headings, carrying heading context into each chunk."""
    chunks, current_heading, current_text = [], "", []
    for line in md.splitlines():
        if line.startswith("#"):                     # new section
            if current_text:
                chunks.append({"heading": current_heading,
                               "text": " ".join(current_text)})
            current_heading = line.strip("# ").strip()
            current_text = []
        elif line.strip():
            current_text.append(line)
    if current_text:
        chunks.append({"heading": current_heading, "text": " ".join(current_text)})
    # merge tiny chunks into previous if too small
    merged = []
    for c in chunks:
        if merged and len(c["text"]) < min_chars:
            merged[-1]["text"] += "\n" + c["heading"] + ": " + c["text"]
        else:
            merged.append(c)
    return merged
```

### Interview
**Q: Chunking ko "structure-aware" hone ka kya matlab?**
**A:** Headings, sections, tables jaise document structure ko respsect karke chunk boundaries set karna, taaki chunk ek logical unit ho, metadata mein section info available ho (jo filtering aur citation madat deta hai).

---

## 2. Metadata Filtering

### Concept
Retrieval se *pehle* ya *saath* metadata-based filters apply karo: `where={"department": "HR", "year": {"$gte": 2025}}`. Isse search space chhota hota hai — precision badhti hai.

### Why?
Bina filter, embedding sirf text similarity dekhta hai. Date/region/department/category jaisi **structured info embedding mein nahi aati**. Filter = "sirf inhi documents mein se dhoondo."

### How it works (ChromaDB)
```python
results = collection.query(
    query_embeddings=[query_vec],
    n_results=5,
    where={                                      # metadata filter
        "$and": [
            {"department": "HR"},
            {"year": {"$in": [2025, 2026]}},
        ]
    },
    where_document={"$contains": "policy"} if keyword_need else None,
)
```

### When to use
- Documents with clear attributes (date, author, department, doc_type).
- Query naturally targets an attribute: "2026 budget", "HR policy".

### Limitations
- Filter too strict → 0 results → recall drops.
- Metadata chaal-that chahiye: har query mein filter lagana hamesha sahi nahi.
- **Balance:** filter pahle laga kay search karo (precision-first) vs search phir filter karke rerank.

### Interview
**Q: Metadata filtering vs embedding-only search ki trade-off?**
**A:** Filtering adds structured signal embedding mein nahi hota — higher precision when attribute known, but risk false-zero recall if filter wrong/too strict, and requires wide metadata you must have planned for ingestion.

---

## 3. Similarity Search Limitations

### Concept
Embedding-based similarity hindi kya karta hai, kya nahi — honest summesh. Advances isi ke aas-pass banti hain.

### Limitation catalog (yehi taali interview ke)
1. **Keyword-insensitive:** "SKU-2045" exact token search nahi hota — vector space me approximate location. Rare/meaningless tokens mishandle.
2. **Negation:** "not eligible" vs "eligible" close vectors ho sakte hain.
3. **Length normalization:** Lambe chunks ka average vector "bland" ho jata hai (similarity diluted).
4. **Out-of-vocabulary entities:** TZ9 product codes ki embeddings meaningless.
5. **Semantic drift:** "bank" bank-of-river vs bank-of-money — context miss.
6. **False closeness:** Sheer topic overlap par relevant lagta hai, exact answer miss karta hai.

### How to compensate
- Hybrid (BM25) for exact/entity.
- Reranking (cross-encoder) grounds similarity with query-aware scoring.
- Query transformation for ambiguity.
- Parent-child for context completeness.

### Interview
**Q: Cosine similarity exact keyword match nahi pakadti — kaise solve karenge?**
**A:** Handler: hybrid/lexical search (BM25/bm25s) entity-heavy docs ke liye; rerankers for precision; plus domain fine-tuned embeddings. Naive cosine koi exact semantics guarantee nahi.

---

## 4. Maximum Marginal Relevance (MMR)

### Concept
**MMR** = diversity-aware selection. Baseline top-k sab similar ho sakte hain (sab same section ka). MMR balance karta hai **relevance** aur **novelty**:

```
MMR = argmax_score [ relevance_i − λ · max_{j ∈ selected} similarity(i, j) ]
```
High λ = more diverse. Low λ = relevance-first.

### Why?
Chunk overlap + same-topic chunks → top-k redundant → waste context, LLC ek hi cheez baar-baar padhta hai. MMR knowingly spread the pick across different ideas.

### Implementation
```python
import numpy as np

def mmr(query_vec, cand_vecs, doc_vecs=None, k=5, lam=0.5):
    if doc_vecs is None:
        doc_vecs = cand_vecs          # for naive MMR, candidates are their own neighbors
    scores = np.array([cos_sim(query_vec, v) for v in cand_vecs])
    selected, cand_idx = [], list(range(len(cand_vecs)))
    while len(selected) < k and cand_idx:
        best = -1; best_i = -1
        for i in cand_idx:
            m = 0.0 if not selected else max(cos_sim(cand_vecs[i], doc_vecs[j]) for j in selected)
            mmr_i = scores[i] - lam * m
            if mmr_i > best:
                best, best_i = mmr_i, i
        selected.append(best_i)
        cand_idx.remove(best_i)
    return selected
```

### Limitations
- λ tuning needed.
- Diversity can push out the single best relevant chunk.
- Extra compute O(k·N).

### Interview
**Q: MMR kya fix karta hai?**
**A:** Top-k redundancy — jab of similar overlapping chunks dominate result, MMR picks diverse highly-relevant mix (relevance − λ·redundancy), better context coverage.

---

## 5. Multi-Query Retrieval

### Concept
Ek query → **LLM se multiple paraphrased/reframed variants** generate karo → har variant se retrieve karo → results merge/dedupe → final context.

### Why?
User query ambiguous/underspecified hoti hai ("Python learning resources"). Ek hi representation se gayi embedding search, alag angles miss kar deti hai. Multi-query = aap khud hi alag interpretation dekho.

### Implementation
```python
from openai import OpenAI
client = OpenAI()

def generate_variants(query: str, n: int = 4) -> list[str]:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content":
            f"Rewrite this search query into {n} different phrasings that "
            f"capture different aspects/perspectives:\n{query}\nReturn only questions, one per line."}],
        temperature=0.7,
    )
    return [l for l in resp.choices[0].message.content.splitlines() if l.strip()]

def multiquery_search(store, query: str, top_k_per: int = 3) -> list[dict]:
    variants = [query] + generate_variants(query)
    results = []
    for q in variants:
        results.extend(store.search(q, top_k_per))
    return dedupe_by_id(results)          # merge by chunk id, keep min distance/rank
```

### Limitations
- LLM call overhead (latency + cost).
- Variants ko dedupe aur budget demanage karna padta hai.
- If queries degrade → noise increase.

### Interview
**Q: Multi-query retrieval kab useful?**
**A:** Ambiguous/vague user queries — multiple phrasings explore different facets of intent, better recall; costs extra LLM calls and requires dedup before context.

---

## 6. Query Transformation

### Concept
Any preprocessing on the user query before retrieval: expansion, Rewards, decomposition, routing, rewriting to a better search form. Combined: "query pipeline".

### Why?
User queries are *conversational*; embeddings are *semantic*. Unclean query (typos, pronouns "ye usage", no keywords) → weak query vector. Transformation = adapt raw query to retrieval form.

### Common transformations
| Transformation | Use-case |
|---|---|
| Rewrite | "uske baare mein" → resolve pronouns, add context |
| Expand | add synonyms, related terms |
| Decompose | multi-part question → sub-queries |
| Translate | cross-language corpus |
| Keywordize | verbose query → concise keywords |
| Route | decide which collection / method to use |

### Implementation — prompt-based rewrite
```python
def rewrite_for_retrieval(query: str, history: list[dict] | None = None) -> str:
    msgs = []
    if history:
        msgs.append({"role": "system", "content":
            "Rewrite the latest user query into a standalone, keyword-rich search query, "
            "resolving pronouns using the conversation history."})
        msgs.extend(history[-4:])
    else:
        msgs.append({"role": "user", "content":
            f"Rewrite for search: {query}\nReturn only the rewritten query."})
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, temperature=0.3)
    return resp.choices[0].message.content.strip()
```

### Interview
**Q: Query transformation vs query expansion — difference?**
**A:** Transformation changes the query *form* (rewrite/decompose/translate); expansion makes existing query *broader* (adds similar terms). Both happen before embedding for better recall.

---

## 7. Query Expansion

### Concept
Query ke saath related synonyms/terms add karo, ya output-based expansion (add top retrieved pseudo-terms). Resulting vector richer, catches more matching chunks.

### Two kinds
1. **Synonym expansion (pre-retrieval):** "car accident → auto collision, vehicle crash, traffic incident."
2. **Pseudo-relevance feedback (post first-pass):** first retrieval se top keywords le kar second round query banao (PRF).

### Implementation — synonym expansion, simple
```python
SYNONYMS = {
    "leave": ["leave policy", "annual leave", "paid time off", "time off", "holiday"],
    "salary": ["compensation", "pay", "remuneration", "salary structure"],
}
def expand_query(query: str, synonyms: dict) -> str:
    tokens = query.lower().split()
    extra = [syn for t in tokens for syn in synonyms.get(t, [])]
    return " ".join(tokens + extra)
```

### Limitations
- Over-expansion → topic drift, low precision.
- Static synonym map breaks in dynamic domains.
- Use expansion with care; evaluate against baseline.

### Interview
**Q: Query expansion drawback kab?**
**A:** Broadens recall but can dilute precision and change intent (multi-sensitive words); not for most operator queries. Synonym construction easily stale in dynamic vocab.

---

## 8. Hybrid Search (BM25 + Vector)

### Concept
Two retrievers combine: **lexical (BM25)** exact-keyword scoring + **semantic (vector)**. Results fused (usually RRF — Reciprocal Rank Fusion) → combined ranking.

```
lexical_score    = BM25(query, chunk)        # exact term overlap (based on IDF)
semantic_score   = cosine(query_vec, chunk)
final_rank       = RRF(rank_lexical, rank_semantic)
```

### Why (biggest story in RAG)
Naive RAG embedding-only does keyword-poor: `SKU-2045`, `HR-104`, product codes, legislation numbers. BM25 exact-match inhe nikal leta hai. Vector catches semantics/phrasing. Dono combine = dono ki strengths, aur failure modes dono ke cancel.

### Practical implementation
```python
from bm25s import BM25               # pip install bm25s
import numpy as np

# Index side
corpus = [chunk_text for chunk_text in all_chunks]
bm25 = BM25()
bm25.index(np.array(corpus))          # simple tokenized build

def hybrid_search(store, query: str, top_k: int = 5, alpha: float = 0.5):
    vec_res = store.search(query, top_k * 2)                # semantic pass
    bm25_res = bm25.retrieve([query], k=top_k * 2)[0]       # lexical pass
    # RRF fusion: score = sum(1 / (60 + rank)) across both lists
    score = {}
    for lst in (vec_ids, bm25_ids):
        for rank, cid in enumerate(lst):
            score[cid] = score.get(cid, 0) + 1 / (60 + rank)
    top_ids = sorted(score, key=score.get, reverse=True)[:top_k]
    return [id_to_chunk[i] for i in top_ids]
```

### Reciprocal Rank Fusion (RRF)
```
RRF(candidate) = Σ over retrievers of (1 / (K + rank))
```
K=60 standard. Simple, robust, no score-normalization needed. That's why hybrid search production favoured.

### Limitations
- Kitne fraction (alpha) tune karna.
- BM25 doesn't understand synonyms (embedding required).
- Extra index + maintenance.

### Interview
**Q: Hybrid search kya optimal hai?**
**A:** Lexical + semantic combine; RRF fuses ranks — exact entity match and semantic paraphrase both covered. Cases where aap sirf keywords expect — hybrid rescue. Tension: resource cost of dual index.

---

## 9. Reranking

### Concept
Reranking: **first-stage** = cheap broad retrieval (top-50), **second-stage** = expensive precise reranker that scores each candidate against the query and **reorders**, keeping top-k.

### Why? (Two-stage retrieval)
Naive: vector DB → directly top-k. Problem: embedding similarity is weak relevance signal. Better: retrieve generously (top 50-100), then use **query-aware model** to recommend few best (top 5-10). Recall preserved by broad first stage; precision restored by rerank.

### Most-common rerankers
- **Cross-encoder** (BAAI/bge-reranker, ms-marco-MiniLM, Cohere Rerank API) — best quality.
- **LLM-as-reranker** — prompt LLM: "rank these passages; output only order" — flexible but slow/costly.
- **Rule-based/tiny models** — fast mid-quality.

### Implementation (Cohere or local cross-encoder)
```python
from sentence_transformers import CrossEncoder           # local
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, candidates: list[tuple[str, dict]], top_k: int = 5):
    pairs = [(query, doc) for doc, _ in candidates]
    scores = reranker.predict(pairs)                     # [-1,1] relevancy scores
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    return [candidates[i] for i in order[:top_k]], scores[order[:top_k]]
```

### Interview
**Q: Retrieve-then-rerank architecture kya?**
**A:** Broad cheap retrieval (top 50) then query-aware cross-encoder/LLM reranking to a few best — up to ~30-50% retrieval quality gain over naive top-k, at cost of extra model call (~ms with cross-encoder, slower for LLM rerank).

---

## 10. Cross-Encoder Rerankers

### Concept
Cross-encoder = model that takes **query and passage TOGETHER** (concatenated) → single relevance score:

```
cross-encoder( query ∥ passage ) → score
```
Contrast with bi-encoder (embedding models): query and passage embedded **separately** (cheap, precomputable, but less precise). Cross-encoder sees full interaction — much better relevance judgment.

### Why cross-encoder over embedding similarity
| | Bi-encoder (embedding) | Cross-encoder (reranker) |
|---|---|---|
| Inputs | 2 separate vectors | query+passage together |
| Compute | cheap, indexable | expensive, per-pair |
| Precision | moderate | high (attention between them) |
| Use | candidate generation | re-ranking |

### Why hierarchical design
You can't cross-encode millions in index — that's why rerank only top-50.

### Interview
**Q: Bi-encoder reasoning vs cross-encoder — where each?**
**A:** Bi-encoder embeddings precompute → scalable first-stage retrieval over all docs; cross-encoder scores query-passage jointly → high precision on small candidate set. Together = best trade-off in production.

---

## 11. Parent-Child Chunking

### Concept
Two-level chunking: **small "child" chunks** (e.g., 200 tokens) embedded & retrieved (precise matches) → retrieve child → **return its parent chunk** (larger context, e.g., 1500 tokens or its section) as context to LLM.

```
child_1 child_2 ... child_n   (small, embedded, searched)
           └──────────┘
       parent = section/paragraph they belong to (bigger context given to LLM)
```

### Why?
Two conflicting needs solved simultaneously:
- **Retrieval precision:** small chunks match precisely.
- **Answer context:** small chunk often doesn't carry enough surrounding info → parent provides it.

### Implementation
```python
def store_parent_child(store, parent_text: str, parent_id: str, meta: dict,
                       child_size=200, overlap=40):
    children = recursive_chunk(parent_text, child_size, overlap)
    store.store_chunks(
        children, [meta | {"parent_id": parent_id, "is_parent": False}] for _ in children,
        [f"{parent_id}-c{i}" for i in range(len(children))])
    store.store_chunks([parent_text], [meta | {"is_parent": True}],
                       [f"{parent_id}-p"])

def retrieve_with_parent(store, query, top_k_children=3):
    children = store.search(query, top_k_children)
    parent_ids = {doc["parent_id"] for doc in children}
    return store.get_by_ids(list(parent_ids))    # fetch full parents
```

### When to use
- Long technical docs (policies, papers) where sections big, but fourprecise answer lies in a sentence.
- Cost/context budget allows parent size.

### Limitations
- Larger context → tokens up.
- Parent definition (section vs paragraph) needs design.

### Interview
**Q: Parent-child chunking twist aapko kuch bhi?**
**A:** Retrieve small child (precise), return parent (context-tolerant) — resolves precision-vs-context tension; redundancy manageable (children in index, parents on-demand).

---

## 12. Sentence-Window Retrieval

### Concept
Embed & search on **individual sentences** (or tiny windows) → once a sentence is retrieved, expand it to a **window of neighboring sentences** for the LLM context.

### Why?
Sentences are the most precise retrieval granularity; but single sentence context-poor. Window (N sentences before/after) provides narrative continuity.

### Implementation
```python
def setup_sentence_index(text: str, store, window: int = 3):
    sents = split_sentences(text)
    for i, s in enumerate(sents):
        win = " ".join(sents[max(0, i-window): i+window+1])   # window context
        store.store_chunks([win], [{"sent_index": i}], [f"w{i}"])
```

### Limitations
- Sentence boundary working with tables/code.
- Window itself embedded → boundaries slightly fuzzy.

### Interview
**Q: Sentence-window vs parent-child?**
**A:** Both "retrieve small, return context." Sentence-window is at sentence granularity (window around it); parent-child at structural sections. Choice depends on structure: prose → sentences; structured sections/tables → parent-child.

---

## 13. Contextual Retrieval (Anthropic)

### Concept
Before embedding, **prepend a small LLM-generated context** to each chunk: e.g., `"This chunk is from Acme's Leave Policy 2026, section 'Earned Leave'..."` Then embed `context + chunk` together. Plus optional BM25 of `context + chunk`. Anthropic reported ~67% retrieval error reduction with context + BM25 hybrid.

### Why?
Chunks in isolation can be **semantically ambiguous** — especially tables (columns values without header context), code (identifier names), fragments ("...in this case 24 days"). Prepended context resolves standalone chunk's meaning.

### Implementation (core idea)
```python
def contextualize_chunk(chunk: str, doc_meta: dict) -> str:
    prompt = f"""
This is a chunk from '{doc_meta['source']}', page {doc_meta['page']}.
Provide a concise 2-4 sentence context that makes this chunk self-explanatory
(e.g., the document, section, and what follows). Keep it factual.

Chunk: {chunk}
Context:"""
    ctx = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=80).choices[0].message.content.strip()
    return f"{ctx}\n\n{chunk}"          # embed & index THIS string
```

### Limitations
- LLM calls per chunk at ingestion (cost/latency; add caching/parallel).
- Context can drift/bias the embedding.
- Storage grows slightly.

### Interview
**Q: Contextual retrieval ('contextual embeddings')?**
**A:** During ingestion, prepend LLM-generated chunk context before embedding — resolves ambiguity of standalone chunks (tables, code, fragments), large retrieval improvements; pairs well with hybrid BM25 on same contextualized text.

---

## 14. Contextual Compression

### Concept
After retrieval, **compress each chunk** to only the relevant parts before the LLM — either extractive (keep relevant sentences) or abstractive (LLM summary). Result: compact context within budget.

### Why?
Chunks bade hotay hain (500 tokens), top-5 = 2500 tokens. Agar sahi answer sirf 200 tokens kaa hai, we're wasting ~2300 tokens on noise — diluting the answer signal and increasing cost/latency. Compression = pass filter, only relevant passes.

### Implementation — extractive LM-based
```python
def compress(query: str, chunk: str, token_budget: int = 150) -> str:
    prompt = (f"Given this query:\n{query}\n\n"
              f"Extract only the sentences relevant to the query from this passage. "
              f"If none are relevant, return '<no_relevant>'.\nPassage:\n{chunk}\n\nRelevant:")
    resp = client.chat.completions.create(model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}], temperature=0)
    out = resp.choices[0].message.content.strip()
    return "" if "<no_relevant>" in out else out   # drop irrelevant chunks entirely
```
Filtering empty → don't include → finals context is compressed & clean.

### Why worthwhile
Combines retrieval with a token-budget lens — reusable across pipelines by hiding the irrelevant.
- Trade-off: extra LLM latency + cost per chunk.
- Explainable vs "ILLM" approaches (LlamaIndex auto-merge) which keep/summarize heuristics.

### Interview
**Q: Contextual compression techniques?**
**A:** Retrieval post-step: extractive (select sentences), LM summarization (abstractive), or filter-by-relevance classifier; keeps LLM input focused on answer-relevant content despite broad retrieval.

---

## Summary: which improvement, which failure

| Failure mode | Improvement to apply |
|---|---|
| Exact keyword / entity miss | Hybrid search (BM25 + vector) |
| Redundant top-k | MMR |
| Ambiguous query | Multi-query + query rewrite/expand |
| Precise sentence in big chunk | Sentence-window, parent-child |
| Standalone chunk ambiguity | Contextual retrieval |
| Context budget overflow + noise | Contextual compression |
| Broad recall then poor precision | Two-stage retrieve + rerank |

---

> **PART 2 continues in `03_advanced_rag_techniques.md`** — wahan 12 techniques har ek ka pura deep-dive (problem/why-naive-fails/internal/architecture/when/advantages/limitations/implementation/example), phir combined advanced pipeline + 20 interview Q&A + exercises + mistakes + decision framework.