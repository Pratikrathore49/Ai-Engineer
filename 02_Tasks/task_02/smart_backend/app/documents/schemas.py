
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentBase(BaseModel):
    """Shared fields returned for a document."""
    filename: str = Field(..., min_length=1, max_length=255)
    file_type: str = Field(..., pattern=r"^(pdf|txt|md)$")
    file_size_bytes: int = Field(..., ge=0)
    status: str = Field(..., min_length=1)


class DocumentUploadResponse(DocumentBase):
    """Response returned after uploading a document."""
    document_id: UUID


class DocumentListItem(DocumentBase):
    """Document summary returned by GET /documents."""
    document_id: UUID
    created_at: datetime


class DocumentDetail(DocumentBase):
    """Detailed document information."""
    document_id: UUID
    created_at: datetime
    chunk_count: int = Field(..., ge=0)
