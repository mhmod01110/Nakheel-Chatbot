from __future__ import annotations

import asyncio

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from nakheel.config import Settings


class QdrantDatabase:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: QdrantClient | None = None

    def connect(self) -> None:
        self.client = QdrantClient(
            host=self.settings.QDRANT_HOST,
            port=self.settings.QDRANT_PORT,
            api_key=self.settings.QDRANT_API_KEY or None,
            https=self.settings.QDRANT_USE_HTTPS,
        )

    def close(self) -> None:
        if self.client is not None:
            self.client.close()

    def ensure_collection(self) -> None:
        assert self.client is not None
        try:
            self.client.get_collection(self.settings.QDRANT_COLLECTION)
            return
        except UnexpectedResponse:
            pass
        self.client.create_collection(
            collection_name=self.settings.QDRANT_COLLECTION,
            vectors_config={
                "dense": VectorParams(
                    size=self.settings.OPENAI_EMBEDDING_DIMENSIONS,
                    distance=Distance.COSINE,
                    on_disk=False,
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
            optimizers_config=OptimizersConfigDiff(indexing_threshold=10000),
        )

    def ping(self) -> bool:
        try:
            assert self.client is not None
            self.client.get_collections()
            return True
        except Exception:
            return False

    def upsert_points(self, points: list) -> None:
        assert self.client is not None
        self.client.upsert(collection_name=self.settings.QDRANT_COLLECTION, points=points)

    async def upsert_points_async(self, points: list) -> None:
        await asyncio.to_thread(self.upsert_points, points)

    def delete_points(self, point_ids: list[str]) -> None:
        assert self.client is not None
        self.client.delete(collection_name=self.settings.QDRANT_COLLECTION, points_selector=point_ids)

    async def delete_points_async(self, point_ids: list[str]) -> None:
        await asyncio.to_thread(self.delete_points, point_ids)

    def dense_search(self, vector: list[float], limit: int):
        assert self.client is not None
        return self.client.query_points(
            collection_name=self.settings.QDRANT_COLLECTION,
            query=vector,
            using="dense",
            limit=limit,
            with_payload=True,
        ).points

    async def dense_search_async(self, vector: list[float], limit: int):
        return await asyncio.to_thread(self.dense_search, vector, limit)

    def sparse_search(self, sparse_weights: dict[int, float], limit: int):
        assert self.client is not None
        return self.client.query_points(
            collection_name=self.settings.QDRANT_COLLECTION,
            query=SparseVector(indices=list(sparse_weights.keys()), values=list(sparse_weights.values())),
            using="sparse",
            limit=limit,
            with_payload=True,
        ).points

    async def sparse_search_async(self, sparse_weights: dict[int, float], limit: int):
        return await asyncio.to_thread(self.sparse_search, sparse_weights, limit)
