# app/documents/routes.py
#
# HTTP endpoints for the documents feature.
#
# WHY UploadFile INSTEAD OF bytes directly?
#   FastAPI's UploadFile streams the file — it doesn't load it all into
#   RAM until you call await file.read(). This lets us check the size
#   before reading and reject oversized files cheaply.

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.auth import verify_api_key
from app.core.db import get_db
from app.documents.processor import extract_text
from app.documents.schemas import (
    DocumentDetail,
    DocumentListItem,
    DocumentUploadResponse,
)
from app.documents.models import Document, DocumentChunk

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    dependencies=[Depends(verify_api_key)],
)

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024   # 10 MB
ALLOWED_EXTENSIONS = {"pdf", "txt", "md"}

def _get_extension(filename: str) -> str:
    """Extract lowercase extension from filename."""
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()

# ── POST /documents/upload ─────────────────────────────────────────

@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile,
    db: Session = Depends(get_db),
):
    """
    Accept a file upload, extract text, save metadata to PostgreSQL.

    Steps:
      1. Validate file extension
      2. Read bytes and validate size
      3. Extract text
      4. Save Document row to DB
      5. Return document_id
    """
    # 1. Validate extension
    ext = _get_extension(file.filename or "")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '.{ext}' not allowed. "
                   f"Upload a pdf, txt, or md file.",
        )

    # 2. Read and validate size
    # WHY read into memory?
    #   We need bytes for text extraction. For files up to 10MB
    #   this is fine. For larger files you'd stream to disk first.
    file_bytes = await file.read()
    file_size = len(file_bytes)

    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size / 1024 / 1024:.1f}MB. "
                   f"Maximum allowed size is 10MB.",
        )

    if file_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info(
        "Document upload received | filename=%s | size_bytes=%d | type=%s",
        file.filename, file_size, ext,
    )

    # 3. Extract text
    # extract_text raises HTTPException on failure — FastAPI catches it
    extracted_text = extract_text(file_bytes, ext)

    logger.info(
        "Text extracted | filename=%s | text_length=%d",
        file.filename, len(extracted_text),
    )

    # 4. Save Document row
    document = Document(
        id=str(uuid.uuid4()),
        filename=file.filename,
        file_type=ext,
        file_size_bytes=file_size,
        status="processing",
        created_at=datetime.now(timezone.utc),
        content=extracted_text
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    logger.info(
        "Document saved | document_id=%s | filename=%s",
        document.id, document.filename,
    )

    return DocumentUploadResponse(
        document_id=document.id,
        filename=document.filename,
        file_type=document.file_type,
        file_size_bytes=document.file_size_bytes,
        status=document.status,
    )

# ── GET /documents ─────────────────────────────────────────────────

@router.get("", response_model=list[DocumentListItem])
def list_documents(db: Session = Depends(get_db)):
    """Return all documents ordered by newest first."""
    documents = (
        db.query(Document)
        .order_by(Document.created_at.desc())
        .all()
    )

    return [
        DocumentListItem(
            document_id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type,
            file_size_bytes=doc.file_size_bytes,
            status=doc.status,
            created_at=doc.created_at,
        )
        for doc in documents
    ]

# ── GET /documents/{document_id} ───────────────────────────────────

@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(document_id: str, db: Session = Depends(get_db)):
    """Return details for one document including chunk count."""
    document = db.query(Document).filter(Document.id == document_id).first()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")

    chunk_count = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .count()
    )

    return DocumentDetail(
        document_id=document.id,
        filename=document.filename,
        file_type=document.file_type,
        file_size_bytes=document.file_size_bytes,
        status=document.status,
        created_at=document.created_at,
        chunk_count=chunk_count,
    )
