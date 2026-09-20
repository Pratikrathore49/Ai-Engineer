# RAG Engineer Learning Guide — Part 3: RAG Evaluation

> **Trainer's Note:** Ab metrics ka stack pata hai. Is file mein: evaluation system kaise design karo — datasets, ground truth, retrieval vs generation eval, RAGAS, human eval, regression testing.

---

## Why RAG Evaluation is Different from Normal LLM Evaluation

Normal LLM eval: prompt → answer quality benchmark (ya human judgment).

RAG eval needs **holistic, multi-stage** view:

| Normal LLM eval | RAG eval |
|---|---|
| Prompt → answer only | Full pipeline: query processing → retrieval → context → generation |
| No retrieval stage | Retrieval metrics independent from generation metrics |
| Single answer quality | Answer quality = relevance × faithfulness × grounding |
| No corpus dependence | Quality depends on corpus, chunking, metadata |
| Static | Any doc update changes eval behavior |

> **Core reason:** In RAG failure kisi bhi stage se aa sakta hai. Sirf answer quality measure karo toh pata nahi chalta *kahan* galati hui. RAG eval stages isolate karta hai taaki fixes targeted hon.

**Interview — Q: RAG eval normal LLM eval se alag kaise?**
A: RAG ek multi-stage system hai. Failure retrieval, context construction, ya generation se ho sakta hai. Agar sirf answer quality dekho toh root cause unclear rahta hai. Isliye RAG eval retrieval metrics (bina LLM) aur generation metrics (LLM) alag-alag measure karta hai, aur both ko corpus/grounding se connect karta hai.

---

## Evaluation Datasets

Har test case ka structure:

```json
{
  "query": "What is the leave policy?",
  "expected_sources": ["leave_policy.pdf", "staff_handbook.pdf"],
  "relevant_chunk_ids": ["c-041", "c-102"],
  "reference_answer": "Employees earn 24 days...",
  "category": "policy_multi_source",
  "difficulty": "medium"
}
```

**Sources of data:**
- Real user logs (sabse valuable)
- Curated domain QA pairs (expert-written)
- LLM-generated synthetic + human validation
- Internal docs ke andar se self-generated questions

**Dataset design rules:**
1. Cover easy / moderate / hard (multi-hop, entity names, negation, ambiguity).
2. Cover failure classes: recall traps, precision traps, no-answer queries, outdated-answer queries.
3. Mix source types (tables, prose, code).
4. Real system ke liye 100-500 golden cases; start 30-50 se.

---

## Ground-Truth Answers

- **Retrieval GT:** relevant chunk ids per query.
- **Generation GT:** reference answer (human expert likha, ya LLM draft + human edit).

**Critical rule:** GT ko apne chunking se *independent* rakho. Gold chunk ids human judgment se map karo (chunker ke output se na lo) — warna tera chunker hi teri eval bias karega, aur badalne par kya hua pata nahi chalega.

---

## Retrieval Evaluation (bina LLM)

```python
def retrieval_metrics(retrieved_ids: list[list[str]], relevant_ids: list[set[str]], k: int = 5):
    hits, mrr_sum = 0, 0.0
    for ret, rel in zip(retrieved_ids, relevant_ids):
        top_k = ret[:k]
        if set(top_k) & rel:
            hits += 1
        for i, cid in enumerate(top_k, start=1):
            if cid in rel:
                mrr_sum += 1.0 / i
                break
    return {
        "hit_rate@k": hits / len(retrieved_ids),
        "mrr@k": mrr_sum / len(retrieved_ids),
        "recall@k": sum(len(set(r[:k]) & rel) / max(1, len(rel)) for r, rel in zip(retrieved_ids, relevant_ids)) / len(retrieved_ids),
        "precision@k": sum(len(set(r[:k]) & rel) / k for r, rel in zip(retrieved_ids, relevant_ids)) / len(retrieved_ids),
    }
```

Run this on the full golden set jab bhi chunking/embedding/query pipeline badalte ho. Baselines store karo, diff karo.

**Interview — Q: Retrieval eval LLM se independent kyun hota hai?**
A: Kyonki want to measure the retriever itself, uncontaminated by generation. Recall/precision/MRR compute sirf ids aur ranks se hoti hain — cheap, deterministic, reproducible. Ye changes compare karne ka cleanest signal hai.

---

## Generation Evaluation

- **Isolate generation:** Retrieval ke results ko *fixed* rakho (same context), sirf generation par measure karo.
- **Metrics:** answer relevance, correctness (vs reference), faithfulness, groundedness.

**Critical practice:** Har generation evaluation ke saath *input context bhi log karo* — bina context ke aap generation ko debug nahi kar sakte.

**Interview — Q: Generation eval mein snapshots kyun log karte hain?**
A: Faithfulness/correctness judge karne ke liye jo context LLM ko diya tha wo record hona chahiye. Wo hi ek source of truth hai — kya system ne sahi info *pakad* li thi, aur kya LLM ne use *correctly* use kiya.

---

## RAGAS (Retrieval-Augmented Generation Assessment) — Concepts

[RAGAS](https://docs.ragas.io) ek open-source framework jo RAG systems ko 4 pillars par measure karta hai:
- **Faithfulness:** answers context se supported hain ya nahi.
- **Answer Relevancy:** answers questions ko address karte hain.
- **Context Precision:** required context relevant & top-ranked hai.
- **Context Recall:** required info jo reference answer mein hai, wo retrieved context mein available hai.

**Workflow:**
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

scores = evaluate(
    dataset,                      # query, contexts=[...], answer, ground_truth
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
# -> per-metric scores over the dataset
```

**Interview — Q: RAGAS ke core metrics?**
A: Faithfulness (answer vs context), answer relevancy (answer vs query), context precision (relevant chunks ranked early), context recall (context covers reference facts). Combine karke RAG quality ka holistic view milta hai.

---

## Automated vs Human Evaluation

| | Automated | Human |
|---|---|---|
| Cost | cheap, scalable | expensive, slow |
| Speed | fast, per-commit possible | slow, periodic |
| Signal | precise metrics, noise-prone (LLM-as-judge bias) | nuanced judgment, gold standard |
| Best for | regression, CI gates, large sweeps | final acceptance, edge cases, creative quality |
| Weakness | LLM-as-judge bias, metric gaming | subjectivity, inconsistency |

**Best practice:** Automated eval har commit/PR par (regression), human eval milestone par (release gate aur adhoc blind spots).

---

## Creating Test Cases

1. **Golden set** — real queries + human GT (retrieval ids + reference answers).
2. **Synthetic set** — corpus se auto-generate hard queries (LLM), human-validate subset.
3. **Negative/hard set** — deliberately tricky: negation ("not eligible"), out-of-domain ("tell me a joke"), numeric exactness, external knowledge.
4. **No-answer set** — queries jin ka jawaab corpus mein nahi hai — "don't know" behavior check.

**Interview — Q: No-answer test set kyun include karte hain?**
A: RAG ka gate-keeper role hai — hallucination tab hota hai jab info nahi milti. No-answer set system ki honesty measure karta hai: kya wo confidently galat bolta hai, kya "I don't know" kehta hai, kya false citations deta hai.

---

## Regression Testing for RAG

RAG systems lootne par / embedding-model swap / prompt change → **unknowingly degrade** ho sakte hain. Regression testing = baseline se compare.

**CI pipeline:**
```
[commit] → ingest sample corpus → run eval suite (30-100 cases)
          → measure: recall@k, faithfulness, answer_relevancy, latency
          → compare vs stored baseline (JSON)
          → pass/fail if deltas exceed threshold → block merge if fail
```

**What regression catches:**
- Embedding model update broke vector space.
- Chunking change reduced recall.
- Prompt tweak broke citation format.
- Retriever param (alpha in hybrid) drifted.

**Interview — Q: RAG regression testing kaise run karte hain?**
A: Versioned golden dataset + versioned baseline metrics (JSON), CI/CD pipeline that reruns eval on every retrieval/prompt/embedding change, delta thresholds (e.g., faithfulness -0.03) block merge. Corpus changes ko version karo taaki "old doc" regressions reproduce hop. 

---

**Next:** `05_hallucination_reduction.md` mein dekhoge — RAG ko hallucination-proof kaise banate hain.