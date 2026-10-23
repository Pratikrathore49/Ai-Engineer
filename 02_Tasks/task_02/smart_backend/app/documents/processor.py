# app/documents/processor.py
#
# Text extraction from uploaded files.
#
# WHY A SEPARATE FILE?
#   Text extraction is pure logic — no FastAPI, no DB, no HTTP.
#   Keeping it separate means you can test it in isolation and
#   reuse it from a background worker later without importing FastAPI.

import io
import logging

from fastapi import HTTPException

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = {"pdf", "txt", "md"}

def extract_text(file_bytes: bytes, file_type: str) -> str:
    """
    Extract plain text from uploaded file bytes.

    Args:
        file_bytes: raw bytes of the uploaded file
        file_type:  "pdf", "txt", or "md"

    Returns:
        Cleaned plain text string

    Raises:
        HTTPException 400 for unsupported file types or extraction errors
    """
    file_type = file_type.lower().strip(".")

    if file_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_type}'. "
                   f"Allowed: {', '.join(sorted(SUPPORTED_TYPES))}",
        )

    try:
        if file_type == "pdf":
            return _extract_pdf(file_bytes)
        else:
            # txt and md are both plain text — just decode
            return _extract_text(file_bytes)

    except HTTPException:
        raise  # re-raise our own errors unchanged
    except Exception as exc:
        logger.error(
            "Text extraction failed | file_type=%s | error=%s",
            file_type, str(exc),
        )
        raise HTTPException(
            status_code=422,
            detail=f"Could not extract text from {file_type} file: {str(exc)}",
        )

def _extract_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    # Import here so the rest of the module works even if pypdf
    # is not installed (useful for tests that don't test PDF).
    try:
        from pypdf import PdfReader
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="pypdf is not installed. Run: pip install pypdf==4.3.1",
        )

    reader = PdfReader(io.BytesIO(file_bytes))

    pages_text = []
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            pages_text.append(page_text)
        else:
            logger.debug("PDF page %d had no extractable text", page_num)

    if not pages_text:
        raise HTTPException(
            status_code=422,
            detail="PDF has no extractable text (may be a scanned image PDF).",
        )

    raw = "\n".join(pages_text)
    return _clean_text(raw)

def _extract_text(file_bytes: bytes) -> str:
    """Decode plain text bytes as UTF-8."""
    try:
        raw = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # Fallback: try latin-1 which accepts every byte value
        raw = file_bytes.decode("latin-1")
        logger.warning("File was not valid UTF-8, decoded as latin-1")

    return _clean_text(raw)

def _clean_text(text: str) -> str:
    """
    Remove noise from extracted text.

    - Null bytes crash some downstream tools
    - Excessive blank lines waste tokens
    - Leading/trailing whitespace is never useful
    """
    # Remove null bytes
    text = text.replace("\x00", "")

    # Collapse 3+ consecutive newlines into 2
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces into one (but preserve newlines)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()
