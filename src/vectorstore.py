"""Firestore Vector Search 백엔드."""
import logging
import uuid

from google.cloud.firestore import AsyncClient
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector

from src.config import Settings
from src.models import DocumentChunk

logger = logging.getLogger(__name__)

_BATCH_SIZE = 500


class VectorStore:
    """Firestore Vector Search 백엔드."""

    EMBEDDING_DIM = 768  # text-multilingual-embedding-002

    def __init__(self, settings: Settings):
        self.project = settings.gcp_project_id
        self.database_id = settings.firestore_database_id
        self.collection_name = settings.firestore_collection_name
        self.client = AsyncClient(project=self.project, database=self.database_id)

    def _collection(self):
        return self.client.collection(self.collection_name)

    async def upsert(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> list[str]:
        """벡터 + 청크 메타데이터를 Firestore에 upsert. 문서 ID 목록 반환."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
            )

        ids: list[str] = []
        batch = self.client.batch()
        batch_count = 0

        for chunk, embedding in zip(chunks, embeddings):
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            doc_ref = self._collection().document(doc_id)

            data: dict = {
                "text": chunk.text,
                "embedding": Vector(list(embedding)),
                "source_file": chunk.source_file,
                "source_type": chunk.source_type,
                "chunk_index": chunk.chunk_index,
            }
            if chunk.page_number is not None:
                data["page_number"] = chunk.page_number
            if chunk.section_title:
                data["section_title"] = chunk.section_title
            if chunk.metadata:
                meta = chunk.metadata
                if meta.get("post_id") is not None:
                    data["post_id"] = meta["post_id"]
                if meta.get("post_title"):
                    data["post_title"] = meta["post_title"]
                if meta.get("year") is not None:
                    data["year"] = meta["year"]
                if meta.get("attachments"):
                    data["attachments"] = meta["attachments"]
                if meta.get("url"):
                    data["url"] = meta["url"]

            batch.set(doc_ref, data)
            batch_count += 1

            if batch_count >= _BATCH_SIZE:
                await batch.commit()
                logger.info("Firestore upsert: committed %d docs", batch_count)
                batch = self.client.batch()
                batch_count = 0

        if batch_count > 0:
            await batch.commit()
            logger.info("Firestore upsert: committed %d docs (final)", batch_count)

        return ids

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        source_type_filter: str | None = None,
        year_filter: int | None = None,
    ) -> list[dict]:
        """Firestore find_nearest를 통한 KNN 검색. id, score, 청크 필드를 가진 dict 리스트 반환."""
        q = self._collection()
        if source_type_filter:
            q = q.where(filter=FieldFilter("source_type", "==", source_type_filter))
        if year_filter is not None:
            q = q.where(filter=FieldFilter("year", "==", year_filter))

        vq = q.find_nearest(
            vector_field="embedding",
            query_vector=Vector(list(query_vector)),
            distance_measure=DistanceMeasure.COSINE,
            limit=top_k,
            distance_result_field="distance",
        )

        results: list[dict] = []
        async for snap in vq.stream():
            d = snap.to_dict() or {}
            distance = d.pop("distance", 0.0)
            d.pop("embedding", None)
            results.append(
                {
                    "id": snap.id,
                    "score": 1.0 - float(distance),  # cosine 유사도
                    **d,
                }
            )
        return results

    async def remove(self, ids: list[str]) -> int:
        """ID로 문서 삭제. 삭제된 개수 반환."""
        if not ids:
            return 0
        removed = 0
        batch = self.client.batch()
        batch_count = 0
        for doc_id in ids:
            batch.delete(self._collection().document(doc_id))
            batch_count += 1
            removed += 1
            if batch_count >= _BATCH_SIZE:
                await batch.commit()
                batch = self.client.batch()
                batch_count = 0
        if batch_count > 0:
            await batch.commit()
        return removed

    async def close(self):
        close_fn = getattr(self.client, "close", None)
        if close_fn is not None:
            result = close_fn()
            if hasattr(result, "__await__"):
                await result
