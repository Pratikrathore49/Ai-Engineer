# RAG Engineer Learning Guide — Part 3: RAG Quality Metrics

> **Trainer's Note:** Part 3 ka core message ek hi: **"Jisko measure nahi kar sakte, usse improve nahi kar sakte."** Part 1 mein RAG bana, Part 2 mein improve kiya. Ab time hai *scientific discipline* ki — metrics define karo, tests build karo, hallucinations ko guard karo. Yahi production engineers ko junior se alag karta hai.

---

## RAG Quality: The Metric Stack

### 1. Retrieval Quality

**Concept:** Retrieval quality = kitne *sahi* chunks retrieve hue (bina LLM ko dekhe). RAG quality ka ~80% yahin decide hota hai.

**Why?** LLM sahi context milne par hi sahi answer deta hai. Poor retrieval = koi bhi generation strategy kaam nahi karegi. Isliye retrieval, generation se *alag* measure karna chahiye.

**How to measure:** Ek **golden evaluation set** chahiye — har query ke liye relevant chunk IDs (ground truth). Phir bina LLM run kiye retriever ke top-k results ko ground truth se compare karo:
- `hit_rate@k`, `precision@k`, `recall@k`, `MRR`

**Interview — Q: Retrieval quality alag se kyu measure karte hain, answer quality se nahi?**
A: Kyonki retrieval errors har downstream stage ko corrupt karte hain — stage isolate karna zaroori hai. Retrieval metrics (recall@k, MRR) cheap aur fast hain, bina LLM ke. Generation metrics tab measure karte hain ki LLM ne good context ke saath kya kiya. Alag fixes: retrieval failure → retriever fix karo; generation failure → prompt/LLM fix karo.

---

### 2. Precision

**Concept:** Retrieved chunks mein se kitne actually relevant hain.

```
precision@k = (# relevant retrieved) / k
```

**Example:** Top-5 retrieved, sirf 3 relevant:
```
precision@5 = 3/5 = 0.6
```

**Why?** High precision = LLM ko irrelevant context kam diya (noise/hallucination risk kam). Irrelevant chunks LLM ko distract/confuse karte hain.

**Interview — Q: Precision kab sabse important?**
A: Jab context noise se answer quality degrade hoti hai (numerical/legal/medical facts) aur jab token cost matter karti hai. Precision-first approach tab chahiye jab bahut noise retrieve karna expensive ho.

**Practice:** Apne Part 1 corpus par top-5 results manually review karo, aur 5 queries ke liye precision@5 calculate karo.

---

### 3. Recall

**Concept:** Sahi chunks mein se kitne retrieve kar liye.

```
recall@k = (# relevant retrieved) / (# total relevant in corpus)
```

**Example:** Corpus mein 4 relevant chunks hain, top-5 mein 3 aaye:
```
recall@5 = 3/4 = 0.75
```

**Why?** Low recall = sabse important info miss ho gayi → LLM ke paas jawab nahi → hallucination ya "don't know" answers.

**Interview — Q: Recall vs precision — trade-off kaise balance karein?**
A: top-k badhao → recall up, precision down. Sweet spot task par depend: exactness zaroori ho → precision; completeness chahiye (multi-fact) → recall. Reranking precision side fix kar deta hai bina recall khoye (retrieve broad, rerank sharp).

---

### 4. Context Relevance

**Concept:** Final context (construction ke baad) query se kitna semantically relevant hai.

**Why?** Top-k raw chunks vs final context block — dedup, compression, ordering se context badal jata hai. Context relevance *final input* measure karta hai jo LLM ko milta hai.

**How:**
- LLM-as-judge: "Given query X, is context C relevant? Score each chunk." → average.
- Ya reference-free: query aur context ke beech embedding similarity.

**Interview — Q: Context relevance vs retrieval precision?**
A: Precision retrieval-time metric hai (chunk ids vs ground truth). Context relevance construction ke *baad* evaluate karta hai — dedup/filtering/compression mein hue mistakes pakadta hai, retriever nahi.

---

### 5. Context Precision

**Concept:** Relevant context chunk kitna *upar* (early) ranked hai.

```
context_precision = Σ(precision@k × relevance_indicator_at_k) / Σ(relevance_indicator_at_k)
```

**Why?** Position matter — LLM early tokens par zyada attention deta hai. Relevant chunk at position 5 → LLM miss kar sakta hai. Good context precision = most pertinent context top par.

**Interview — Q: Context precision kaise improve karein?**
A: Reranking (cross-encoder) jo relevance ke hisaab se reorder karta hai — directly context precision improve karta hai. Relevant tokens early mein hone se miss hone ka risk kam ho jata hai.

---

### 6. Context Recall

**Concept:** Ground-truth answer kitna context se cover ho pata hai — "retrieved context se ye answer derive ho sakta hai?"

**Why?** Chunk retrieve hua, lekin key sentence missing ho sakta hai (chunk boundary, compression drop). Context recall = coverage check.

**How:**
```
For ground-truth statement S in reference_answer:
    can S be attributed to any retrieved chunk?   (LLM/NLI check)
context_recall = attributable_statements / total_statements
```

**Interview — Q: Context recall low hone par kya check karein?**
A: Chunking granularity (info split thin/thick), top-k too low, compression over-aggressive (needed sentences drop), filters needed docs exclude kar rahe hain. Ye saare symptoms hain.

---

### 7. Answer Relevance

**Concept:** Generated answer user ka question actually address karta hai ya nahi (correctness ignore, sirf topic match).

**Why?** Answer plenty ho sakta hai but unrelated (dusty response). Relevance="kya sahi sawal ka jawab diya", faithfulness="jo bola wo context se supported hai ya nahi".

**How:** LLM-judge compare answer vs query. Score 0-1.

**Interview — Q: Answer relevance vs faithfulness — difference?**
A: Relevance = kya question ko address karta hai (right topic); faithfulness = kya statements context se supported hain (no fabrication). Answer fully relevant ho sakta hai phir bhi hallucinated, ya faithful ho par off-topic. RAG mein DONO chahiye.

---

### 8. Faithfulness

**Concept:** Answer ke statements kitne retrieved context se supported hain. **Core hallucination metric.**

```
faithfulness = (# supported statements) / (# total statements)
```

**How measured (LLM-based):**
```
1. Decompose answer into atomic claims/statements.
2. Har statement: "Is this supported by the context?" → yes/no
3. faithfulness = supported / total
```

**Why?** Faithfulness low = LLM apni training knowledge se bana raha hai, context se nahi. Ye RAG system ka failure hai (retrieval/context/prompt ne allow kiya) — sirf LLM ki galti nahi.

**Interview — Q: Faithfulness kaise calculate hota hai?**
A: Answer ko independent claims mein decompose karo, har claim ko retrieved context se verify karo (LLM ya NLI model), ratio supported/total. RAGAS isi pattern use karta hai.

---

### 9. Groundedness

**Concept:** Faithfulness ka strict cousin — answer statements *actually present* hona specific source mein (citation-verifiable).

**Groundedness vs faithfulness — nuance:**
- Faithfulness: statements context se supported.
- Groundedness: statements specific source snippet se directly attributable (token-level, citation checks).

**Why it matters:** Production mein grounded answers sources cite karte hain — users verify kar sakte hain.

**Interview — Q: "Provide citations" kyun groundedness improve karta hai?**
A: Model ko `[Source N]` emit karne se majboor karta hai ki har claim kaha se aaya sochkar bataye — unsupported assertions reduce hote hain, aur humans ko verifiable trace milta hai. Weak-lekin-real grounding pressure.

---

### 10. Hallucination Detection

**Concept:** Runtime detection ki answer fabricated content hai — baad mein janchna nahi, *real-time* guard.

**Detection strategies:**

| Strategy | Mechanism | Cost |
|---|---|---|
| Distance/score threshold | retrieval distance too high → "no info" declare | free |
| LLM faithfulness check | answer claims verified against context on the fly | 1 LLM call |
| NLI model | `entailed/contradicted/neutral` classify sentence+context | fast local model |
| N-gram overlap | answer vs context token overlap ratio | free, weak |
| Model confidence | too-low confidence → suspect | model-specific |

**Post-generation guard:**
```
1. Generate answer.
2. Run claims-support check / NLI.
3. Unsupported claim: drop it / hedge it / return "not in docs".
4. Return answer + support flags. Har failure log karo → eval set mein add.
```

**Interview — Q: Real-time hallucination detection approach?**
A: Retrieval-distance precheck (cheap) + post-generation NLI/LLM faithfulness verification. Unsupported claims remove/flag karo; har detected case eval set mein add kar ke retrieval fixes chalao. Detection sirf filter nahi, ek monitor hai.

---