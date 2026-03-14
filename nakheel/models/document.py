from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Lifecycle states for an indexed knowledge source."""

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentSourceType(str, Enum):
    """Supported document ingestion origins."""

    PDF = "pdf"
    COPIED_DOC = "copied_doc"


class DocumentMetadata(BaseModel):
    """Stored metadata for a source document in MongoDB."""

    doc_id: str
    filename: str
    source_type: DocumentSourceType = DocumentSourceType.PDF
    title: str | None = None
    language: str = "mixed"
    total_pages: int = 0
    total_chunks: int = 0
    file_size_kb: float = 0
    uploaded_at: datetime
    indexed_at: datetime | None = None
    status: DocumentStatus = DocumentStatus.PENDING
    qdrant_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    description: str | None = None
    current_step: str | None = None
