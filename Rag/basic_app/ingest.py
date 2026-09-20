"""Document loading + ingestion.

Runs once per corpus/update: PDF -> text -> clean -> chunks -> embed -> store.
"""

import pathlib
import uuid

import pypdf

from chunking import recursive_chunk
from config import settings
from vector_store import VectorStore


def load_pdf(file_path: pathlib.Path) -> list[dict]:
    """Parse a PDF into per-page text blocks.

    Input : path to a PDF.
    Output: [{"page": int, "text": str}, ...]
    """
    reader = pypdf.PdfReader(str(file_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": i, "text": text})
    return pages


def clean_text(text: str) -> str:
    """Remove common scraper noise, collapse whitespace."""
    text = " ".join(text.split())          # normalize all whitespace
    text = text.replace("-\n", "")          # fix hyphenated line breaks
    return text.strip()


def ingest_pdf(store: VectorStore, file_path: pathlib.Path) -> int:
    """Load -> clean -> chunk -> embed -> index a single PDF. Returns chunk count."""
    pages = load_pdf(file_path)
    doc_id = str(uuid.uuid4())[:8]
    chunk_ids, chunk_texts, chunk_metas = [], [], []

    for block in pages:
        text = clean_text(block["text"])
        if not text:
            continue
        for idx, chunk in enumerate(recursive_chunk(
                text, settings.chunk_size, settings.chunk_overlap)):
            chunk_ids.append(f"{doc_id}-{block['page']}-{idx}")
            chunk_texts.append(chunk)
            chunk_metas.append({
                "doc_id": doc_id,
                "source": file_path.name,
                "page": block["page"],
                "chunk_index": idx,
                "embedding_model": settings.embedding_model,
            })

    store.store_chunks(chunk_texts, chunk_metas, chunk_ids)
    return len(chunk_texts)


def ingest_all() -> int:
    """Ingest every PDF in data/docs/. Memory of stored docs for re-indexing.""" 
    store = VectorStore()
    total = 0
    for path in sorted(pathlib.Path(settings.data_dir).glob("*.pdf")):
        n = ingest_pdf(store, path)
        total += n
        print(f"  {path.name}: {n} chunks")
    return total


if __name__ == "__main__":
    total = ingest_all()
    store = VectorStore()
    print(f"Done. Stored {total} chunks across {store.collection.count()} entries.")