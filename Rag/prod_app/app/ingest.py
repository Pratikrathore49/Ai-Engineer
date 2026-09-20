"""Ingestion pipeline: upload/read PDF -> parse -> clean -> chunk -> embed -> index.

Production hardening compared to Part 1:
- validation (extension, size)
- per-document doc_id for clean re-index/delete (upsert semantics)
- metadata carries freshness + version for eval/audit
"""

import pathlib
import uuid
from dataclasses import dataclass

import pypdf

from app.bm25_index import BM25
from app.chunking import recursive_chunk
from app.config import settings
from app.logging_conf import get_logger
from app.vector_store import VectorStore

logger = get_logger("ingest")

ALLOWED_EXTENSIONS = {".pdf"}


class IngestionError(Exception):
    pass


def validate_upload(filename: str, size_bytes: int) -> None:
    ext = pathlib.Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise IngestionError(f"Unsupported file type {ext}. Allowed: {sorted(ALLOWED_EXTENSIONS)}")
    if size_bytes > settings.max_upload_mb * 1024 * 1024:
        raise IngestionError(f"File too large: max {settings.max_upload_mb} MB")


def parse_pdf(path: pathlib.Path) -> list[dict]:
    """PDF -> [{"page": int, "text": str}]."""
    reader = pypdf.PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": i, "text": text})
    return pages


def clean_text(text: str) -> str:
    text = " ".join(text.split())
    text = text.replace("-\n", "")
    return text.strip()


@dataclass
class IngestResult:
    doc_name: str
    chunk_count: int
    doc_id: str


def ingest_document(store: VectorStore, file_path: pathlib.Path) -> IngestResult:
    doc_id = str(uuid.uuid4())[:8]
    pages = parse_pdf(file_path)
    ids, texts, metas = [], [], []

    for block in pages:
        text = clean_text(block["text"])
        if not text:
            continue
        for idx, chunk in enumerate(recursive_chunk(text, settings.chunk_size, settings.chunk_overlap)):
            ids.append(f"{doc_id}-p{block['page']}-{idx}")
            texts.append(chunk)
            metas.append({
                "doc_id": doc_id,
                "source": file_path.name,
                "page": block["page"],
                "chunk_index": idx,
                "embedding_model": settings.embedding_model,
                "ingested_at": pathlib.Path(file_path).stat().st_mtime,
            })

    if not texts:
        raise IngestionError(f"{file_path.name}: no text extracted (scanned PDF? OCR needed)")

    store.add_chunks(ids, texts, metas)
    logger.info("ingested doc=%s chunks=%d doc_id=%s", file_path.name, len(texts), doc_id)
    return IngestResult(doc_name=file_path.name, chunk_count=len(texts), doc_id=doc_id)


def rebuild_bm25(store: VectorStore) -> BM25:
    """Rebuild lexical index from current collection (call after ingestion)."""
    _, _, documents = store.get_all_chunks()
    bm25 = BM25()
    bm25.index(documents)
    logger.info("rebuilt BM25 over %d chunks", len(documents))
    return bm25