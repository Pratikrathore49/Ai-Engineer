"""The RAG brain: retrieve -> build context -> prompt -> LLM generation.

Every query flows through here. Kept deliberately thin: each step is one
function call so you can log/debug each stage independently.
"""

from openai import OpenAI

from config import settings
from vector_store import VectorStore

SYSTEM_TEMPLATE = """You are a helpful assistant for {company}.
Answer ONLY using the provided context below.
If the context does not contain the answer, say exactly:
"I don't have this information in the provided documents."
Always cite sources in the format [Source N] right after the supporting sentence.
Be concise and factual. Max {max_words} words."""


def build_context(search_result: dict, max_chars: int = 4000) -> tuple[str, list[dict]]:
    """Turn the search response into a labeled context block.

    Input : raw Chroma query result.
    Output: (formatted context string with [Source N] labels,
             list of source metadata for citations).
    """
    parts, sources, used = [], [], 0
    docs = search_result["documents"][0]
    metas = search_result["metadatas"][0]
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        if used + len(doc) > max_chars:
            break
        label = f"{meta.get('source', '?')} page {meta.get('page', '?')}"
        parts.append(f"[Source {i}: {label}]\n{doc}")
        sources.append(meta)
        used += len(doc)
    return "\n\n".join(parts), sources


class RAGEngine:
    def __init__(self, store: VectorStore) -> None:
        self.store = store
        self.llm = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    def answer(self, query: str) -> dict:
        """End-to-end inference. Input: user query. Output: answer + sources."""
        search = self.store.search(query, settings.top_k)
        if not search["documents"][0]:
            return {"answer": "No relevant documents found.", "sources": []}

        context, sources = build_context(search)

        system = SYSTEM_TEMPLATE.format(company="Acme Corp", max_words="200")
        user = f"Context:\n\n{context}\n\nQuestion:\n{query}\n\nAnswer:"

        response = self.llm.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": sources,
            "distance_scores": search["distances"][0],
        }