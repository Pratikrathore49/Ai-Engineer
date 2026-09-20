# RAG Engineer Learning Guide — Part 3: Hallucination Reduction

> **Trainer's Note:** Hallucination reduction = ek single trick nahi, **layered defense**. Layer 1: system se galat info nikdo hi na (retrieval). Layer 2: jo context diya wo LLM ko shoehorn karne na de (prompt). Layer 3: LLM ne kya bola — verify karo (validation). Ye "defense in depth" architecture hai.

---

## 1. Better Retrieval

**Concept:** Sabase powerful hallucination-reduction: sahi chunks provide karo.

**Why?** Jo context mein nahi hai, wo LLM fabricate karta hai. Sahi chunks mil gaye → hallucination ka material hi nahi.

**How:**
- Hybrid search (BM25 + vector) — entity-heavy queries cover.
- Reranking — top-k mein relevant chunks hi aaye.
- Multi-query/rewrite — ambiguous query ke saare angles explore.
- Metadata filters — irrelevant corpus sections exclude.

**Interview — Q: Retrieval hallucination reduction mein sabse strong kyon?**
A: Kyonki hallucination ka root cause mostly hai "LLM ko sahi info mili hi nahi." Retrieval quality improve karo (hybrid, rerank, filters) toh context mein safe info hoti hai — generation ke paas fabricate karne ki zaroorat nahi.

---

## 2. Better Chunking

**Concept:** Sahi granularity + structure-aware chunking.

**Why?** Galat chunking se key info slice/ghoom jati hai, ya chhota chunk context-less reh jata hai → LLM ko incomplete context → guess karna padta hai.

**How:**
- Parent-child / sentence-window: precise retrieval + full context.
- Overlap: boundaries par info loss mat hone do.
- Structure-aware (headings/tables): semantic unit intact.

**Interview — Q: Chunking hallucination ko directly kaise affect karta hai?**
A: Poor chunk placement = relevant info scene mein aana hi nahi → LLM ke paas kho-khaali context jispar wo fill-in-the-blanks karta hai (hallucination). Sahi chunking context-completeness garanti deta hai.

---

## 3. Context Filtering

**Concept:** Retrieval ke baad context se irrelevant chunks hatao / low-confidence chunks drop karo.

**Why?** Context mein noise = LLM confusion. Contradicting ya unrelated chunks LLM ko garble kar dete hain.

**How:**
- Score/distance threshold: too-high distance chunks exclude.
- LLM/classifier relevance gate: sirf relevant chunks context mein.
- CRAG-style: bad retrieval detect karke fallback/search.

**Interview — Q: Context filtering thresholds kis tarah set karte hain?**
A: Golden set par distance distribution dekh kar: relevant vs irrelevant chunks ke score overlap identify karo, threshold sweet spot se set karo. Hard threshold risky hai (recall drop) — "soft" approach: LLM relevance gate ya compression.

---

## 4. Reranking

**Concept:** Retrieval ke baad true relevance par reorder — top-k mein na aane wale sahi chunks ko uper lao.

**Why?** Vector similarity ordering weak hai — sahi answer rank 12 par akar top-5 cut se bahar ho sakta hai.

**How:** Two-stage: retrieve top-30-100 → cross-encoder/LLM rerank → top-3-5 only relevant.

**Interview — Q: Rerank hallucination is example?**
A: Jab query "leave carry-forward limit" ho aur top-5 mein 4 chunks leave ke hain par limit wala chunk #7 par hai — LLM ko carry-forward ka exact number nahi milta → wo guess karta hai. Rerank wale chunk ko top par la ke exact answer provide karta hai — hallucination prevented.

---

## 5. Prompt Constraints

**Concept:** System prompt mein hard constraints — answer sirf context se, unknown ko "don't know" bolo.

**Template:**
```
Answer ONLY using the provided context.
If the context does not contain the answer, say exactly:
"I don't have this information in the provided documents."
NEVER use your own knowledge for facts not in the context.
Cite [Source N] for every factual statement.
```

**Why?** "Sirf context" + "don't-know fallback" tera primary verbal guard hai.

**Interview — Q: Prompt constraints hallucination ko fully prevent karte hain?**
A: Nahi — wo reduce karte hain, guarantee nahi. LLM instruction-following imperfect hai (jo search/web/system translation cetai). Isliye prompt constraints hamesha validation layer ke saath combine hone chahiye.

---

## 6. Citation / Grounding

**Concept:** Har fact ke saath source citation. LLM ko har claim ke source pe vibe karo.

**How:**
```
Answer format: each sentence ends with [Source N].
Sources (numbered) are printed after the answer.
```
Output structured as JSON when strict:
```json
{"answer": "...", "citations": [{"source": "file.pdf", "page": 2, "chunk_id": "c-102"}]}
```

**Why?** Citations (1) model ko grounded rehne ke liye pressure dete hain, (2) users verify kar sakte, (3) eval aur logging mein trace milta hai.

**Interview — Q: Production citation design best?**
A: Structured output (JSON) with chunk ids + page + source — tools/UI parsed kar sakte, eval automated verify kar sakta (chunk support check), aur audit trail complete rahta hai. Free-text `[Source N]` readers ke liye okay, automation ke liye weak.

---

## 7. "I Don't Know" Behavior

**Concept:** Context insufficient ho toh model ko honest refusal sikhana.

**Two levels:**
1. **Prompt level:** "if not in context, say X" — instruction.
2. **System level (circuit-breaker):** distance/score threshold check retrieval se pehle → threshold se zyada → LLM call hi mat karo, "no info" respond karo (save cost + hallucination).

**System-level implementation:**
```python
def answer_safe(query: str) -> dict:
    res = store.search(query, top_k)
    if res["distances"][0][0] > settings.min_score_threshold:   # weak match
        return {"answer": "I don't have this information in the provided documents.",
                "sources": [], "mode": "no_retrieval"}
    ...normal pipeline...
```

**Interview — Q: "I don't know" ser system par kaise enforce karte?**
A: Dual: prompt instruction + retrieval circuit-breaker (threshold). Threshold golden-set score distribution se tune hua. LLM-call skip se latency & cost bhi bachti hai aur hallucination root-cause eliminate hota hai.

---

## 8. Confidence / Relevance Thresholds

**Concept:** Retrieval distance/score ko confidence signal use karo.

- Cosine distance close to 0 = strong match.
- Threshold on retrieved top score + context density.

**Care:**
- Distance **threshold too strict** → too many "don't know" (over-refusal).
- Strictness depends on metric embedding model ki — calibrate per model.

**Interview — Q: Threshold calibration?**
A: Golden set se relevant vs irrelevant chunks ke score distributions plot karo; threshold aisi jagah jo golden recall rate preset pe rakhe while keeping false-accept low. Recalibrate jab embedding model change ho.

---

## 9. Detecting Insufficient Context

**Concept:** "Context mein jawaab hai hi nahi" detect karo generation se pehle ya baad.

**Pre-generation:**
- Distance thresholds (upar jaise).
- LLM gate: "Is the answer present in the context? Yes/No with confidence."

**Post-generation (checking-self-correct):**
- Faithfulness check (RAGAS-style claims).
- NLI entailed check: each answer sentence vs context.
- Citation validity: what does `[Source N]` point to, does it contain the cited claim?

**Implementation (LLM faithfulness check):**
```python
def check_faithfulness(answer: str, context: str) -> tuple[float, list[str]]:
    prompt = f"""Check if each claim in the answer is supported by the context.
Answer:\n{answer}\n\nContext:\n{context}
For each sentence of the answer, output SUPPORTED or UNSUPPORTED, one per line."""
    result = llm_fn(prompt, temperature=0)
    lines = [l.strip().upper() for l in result.strip().splitlines()]
    supported = sum(1 for l in lines if l.startswith("SUPPORTED"))
    return supported / max(1, len(lines)), lines
```

---

## Hallucination Defense — Layered Summary

```
[Layer 1 — Input]
  Better retrieval (hybrid, rerank, filters)
  Better chunking (parent-child, sentence-window)
      ↓
[Layer 2 — Context]
  Context filtering (thresholds, relevance gate)
  Contextual compression (noise removal)
      ↓
[Layer 3 — Prompt]
  "Only context" + citation format + don't-know instruction
  temperature low (0-0.3)
      ↓
[Layer 4 — Validation]
  Circuit-breaker (no-info respond)
  Post-generation faithfulness/NLI check
  Citation validity audit
      ↓
[Layer 5 — Feedback]
  Har detected hallucination → eval set mein add → retrieval/prompt fix
```

---

**Next:** `06_production_architecture.md` — in defense layers ko production FastAPI architecture mein kaise slot karte hain.