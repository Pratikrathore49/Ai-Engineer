"""Offline evaluation suite: run retrieval + generation metrics on a golden set.

Usage:
    python -m app.eval.run --dataset app/evals/dataset.jsonl

Metrics:
  - retrieval: hit_rate@k, recall@k, precision@k (no LLM needed)
  - generation: faithfulness (LLM-based), answer relevance (LLM-based)
"""

import argparse
import json

from app.config import settings
from app.generation import Generation
from app.logging_conf import get_logger
from app.retrieval import HybridRetriever
from app.vector_store import VectorStore

logger = get_logger("eval")


def load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def eval_retrieval(retriever: HybridRetriever, cases: list[dict], k: int = 5) -> dict:
    hits = recall = prec = 0
    for case in cases:
        retrieved = [h["chunk_id"] for h in retriever.search(case["query"], top_k=k)]
        got = set(retrieved) & set(case["relevant_chunk_ids"])
        if got:
            hits += 1
        recall += len(got) / max(1, len(case["relevant_chunk_ids"]))
        prec += len(got) / k
    n = len(cases)
    return {"hit_rate@k": round(hits / n, 3), "recall@k": round(recall / n, 3),
            "precision@k": round(prec / n, 3)}


def eval_generation(generation: Generation, cases: list[dict], retriever: HybridRetriever, k: int = 5) -> dict:
    faithfulness_scores, relevance_scores = [], []
    for case in cases:
        hits = retriever.search(case["query"], top_k=k)
        context = "\n\n".join(f"[Source {i}] {h['text']}" for i, h in enumerate(hits, 1))
        answer = generation.generate(case["query"], context)
        faithfulness_scores.append(generation.check_faithfulness(answer, context))
        relevance_scores.append(_llm_relevance(generation, case["query"], answer, case.get("reference_answer")))
    return {
        "faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 3),
        "answer_relevance": round(sum(relevance_scores) / len(relevance_scores), 3),
    }


def _llm_relevance(generation: Generation, query: str, answer: str, reference: str | None) -> float:
    prompt = (
        "Score how well the ANSWER addresses the QUESTION from 0 to 1 "
        "(relevant but possibly incorrect = still high). Output only the number.\n\n"
        f"QUESTION: {query}\nANSWER: {answer}"
        + (f"\nREFERENCE: {reference}" if reference else "")
    )
    resp = generation.client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
    )
    text = (resp.choices[0].message.content or "0").strip()
    try:
        return max(0.0, min(1.0, float(text)))
    except ValueError:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="app/evals/dataset.jsonl")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    store = VectorStore()
    retriever = HybridRetriever(store)
    retriever.rebuild_lexical()
    gen = Generation()
    cases = load_dataset(args.dataset)

    print("RETRIEVAL:", eval_retrieval(retriever, cases, args.k))
    print("GENERATION:", eval_generation(gen, cases, retriever, args.k))


if __name__ == "__main__":
    main()