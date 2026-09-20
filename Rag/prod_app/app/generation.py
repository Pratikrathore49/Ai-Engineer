"""Generation with guardrails.

- prompt construction (system + context + question)
- LLM call
- faithfulness check (post-generation)
- insufficient-context / confidence signals
"""

from openai import OpenAI

from app.config import settings
from app.logging_conf import get_logger

logger = get_logger("generation")

SYSTEM_TEMPLATE = """You are a helpful assistant for {company}.
Answer ONLY using the provided context below.
If the context does not contain the answer, say exactly:
"I don't have this information in the provided documents."
NEVER use your own knowledge for facts not present in the context.
For every factual statement, cite the source inline as [Source N].
Keep the answer concise. Max {max_words} words."""


class Generation:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

    def generate(self, query: str, context: str) -> str:
        system = SYSTEM_TEMPLATE.format(company="Acme Corp", max_words="200")
        user = f"Context:\n\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        resp = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        return resp.choices[0].message.content or ""

    def check_faithfulness(self, answer: str, context: str) -> float:
        """Fraction of answer sentences supported by context (0..1)."""
        prompt = (
            "For EACH sentence of the answer below, reply with one token: "
            "SUPPORTED or UNSUPPORTED, deciding whether that sentence is backed "
            "by the context.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"ANSWER:\n{answer}\n\n"
            "Judgements (one per line):"
        )
        resp = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        lines = [l.strip().upper() for l in (resp.choices[0].message.content or "").splitlines() if l.strip()]
        supported = sum(1 for l in lines if l.startswith("SUPPORTED"))
        return supported / max(1, len(lines)) if lines else 0.0

    def insufficient_context(self, hits: list[dict]) -> bool:
        """Circuit-breaker: best hit too weak -> refuse to hallucinate."""
        if not hits:
            return True
        return hits[0]["distance"] > settings.min_relevance_score