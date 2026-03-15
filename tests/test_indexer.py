from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from nakheel.config import Settings
from nakheel.core.ingestion.indexer import DocumentIndexer
from nakheel.models.chunk import Chunk
from nakheel.models.document import DocumentSourceType, DocumentStatus


class RecordingCollection:
    def __init__(self, fail_on_insert: bool = False, fail_on_audit: bool = False):
        self.documents: dict[str, dict] = {}
        self.chunks: list[dict] = []
        self.audit_logs: list[dict] = []
        self.fail_on_insert = fail_on_insert
        self.fail_on_audit = fail_on_audit

    async def insert_one(self, payload):
        if payload.get("event") == "document_indexed":
            if self.fail_on_audit:
                raise RuntimeError("audit failed")
            self.audit_logs.append(payload)
            return None
        self.documents[payload["doc_id"]] = dict(payload)
        return None

    async def insert_many(self, payloads):
        if self.fail_on_insert:
            raise RuntimeError("chunk insert failed")
        self.chunks.extend(dict(item) for item in payloads)
        return None

    async def update_one(self, filters, update):
        doc = self.documents.setdefault(filters["doc_id"], {"doc_id": filters["doc_id"]})
        doc.update(update["$set"])
        return None


class FakeMongo:
    def __init__(self, fail_on_chunk_insert: bool = False, fail_on_audit: bool = False):
        self.collections = {
            "documents": RecordingCollection(),
            "chunks": RecordingCollection(fail_on_insert=fail_on_chunk_insert),
            "audit_logs": RecordingCollection(fail_on_audit=fail_on_audit),
        }

    def collection(self, name: str):
        return self.collections[name]


class FakeQdrant:
    def __init__(self):
        self.upserted = []
        self.deleted = []

    async def upsert_points_async(self, points):
        self.upserted.extend(points)

    async def delete_points_async(self, point_ids):
        self.deleted.append(list(point_ids))


class FakeParser:
    async def parse_to_markdown_async(self, pdf_path: Path, output_path: Path | None = None):
        return "# Title\n\nNew Valley Governorate information for testing.", 2


class FakeEmbedder:
    async def embed_texts_async(self, texts):
        return [[float(index + 1)] * 4 for index, _ in enumerate(texts)]


class FakeSparseEmbedder:
    def fit_transform(self, texts):
        return [{1: 0.5 + index} for index, _ in enumerate(texts)]


def build_indexer(*, fail_on_chunk_insert: bool = False, fail_on_audit: bool = False):
    mongo = FakeMongo(fail_on_chunk_insert=fail_on_chunk_insert, fail_on_audit=fail_on_audit)
    qdrant = FakeQdrant()
    settings = Settings(TEMP_DIR=Path("./tmp/test-indexer"))
    indexer = DocumentIndexer(
        settings=settings,
        mongo=mongo,
        qdrant=qdrant,
        parser=FakeParser(),
        dense_embedder=FakeEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
    )
    chunk = Chunk(
        chunk_id="chk-1",
        doc_id="doc-1",
        chunk_index=0,
        section_title="Title",
        parent_section=None,
        text="New Valley Governorate information for testing.",
        text_ar=None,
        language="en",
        page_numbers=[],
        token_count=8,
        char_count=46,
        overlap_prev=None,
        overlap_next=None,
        created_at=datetime.now(UTC),
    )
    indexer.chunker = SimpleNamespace(chunk_markdown=lambda _markdown, _doc_id: [chunk])
    return indexer, mongo, qdrant


@pytest.mark.asyncio
async def test_indexer_tracks_phase_progress():
    indexer, mongo, _qdrant = build_indexer()
    await indexer._create_document_record(
        doc_id="doc-1",
        batch_id="batch-1",
        filename="phase.pdf",
        source_type=DocumentSourceType.PDF,
        title=None,
        description=None,
        tags=[],
        file_size_kb=1.0,
    )
    seen_steps = []

    async def progress(step: str):
        seen_steps.append(step)

    result = await indexer._index_text_content(
        doc_id="doc-1",
        batch_id="batch-1",
        filename="phase.pdf",
        source_type=DocumentSourceType.PDF,
        raw_text="# Title\n\nNew Valley Governorate information for testing.",
        title=None,
        description=None,
        tags=[],
        language_hint="auto",
        file_size_kb=1.0,
        total_pages=2,
        started=0.0,
        progress_callback=progress,
    )

    assert seen_steps == ["chunking", "embedding", "persisting"]
    assert result["status"] == "indexed"
    assert mongo.collection("documents").documents["doc-1"]["current_step"] == "indexed"


@pytest.mark.asyncio
async def test_indexer_rolls_back_qdrant_when_chunk_persist_fails():
    indexer, mongo, qdrant = build_indexer(fail_on_chunk_insert=True)
    await indexer._create_document_record(
        doc_id="doc-1",
        batch_id=None,
        filename="rollback.pdf",
        source_type=DocumentSourceType.PDF,
        title=None,
        description=None,
        tags=[],
        file_size_kb=1.0,
    )

    with pytest.raises(RuntimeError, match="chunk insert failed"):
        await indexer._index_text_content(
            doc_id="doc-1",
            batch_id=None,
            filename="rollback.pdf",
            source_type=DocumentSourceType.PDF,
            raw_text="# Title\n\nNew Valley Governorate information for testing.",
            title=None,
            description=None,
            tags=[],
            language_hint="auto",
            file_size_kb=1.0,
            total_pages=2,
            started=0.0,
        )

    assert qdrant.deleted == [["chk-1"]]
    assert mongo.collection("documents").documents["doc-1"]["status"] == DocumentStatus.FAILED.value


@pytest.mark.asyncio
async def test_indexer_keeps_indexed_status_when_audit_logging_fails():
    indexer, mongo, qdrant = build_indexer(fail_on_audit=True)
    await indexer._create_document_record(
        doc_id="doc-1",
        batch_id=None,
        filename="audit.pdf",
        source_type=DocumentSourceType.PDF,
        title=None,
        description=None,
        tags=[],
        file_size_kb=1.0,
    )

    with pytest.raises(RuntimeError, match="audit failed"):
        await indexer._index_text_content(
            doc_id="doc-1",
            batch_id=None,
            filename="audit.pdf",
            source_type=DocumentSourceType.PDF,
            raw_text="# Title\n\nNew Valley Governorate information for testing.",
            title=None,
            description=None,
            tags=[],
            language_hint="auto",
            file_size_kb=1.0,
            total_pages=2,
            started=0.0,
        )

    assert qdrant.deleted == []
    assert mongo.collection("documents").documents["doc-1"]["status"] == DocumentStatus.INDEXED.value
