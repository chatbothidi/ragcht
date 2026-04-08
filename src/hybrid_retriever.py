from src.bm25_index import BM25Index
from src.chunk_store import ChunkStore
from src.config import Settings
from src.embeddings import EmbeddingService
from src.vectorstore import VectorStore


class HybridRetriever:
    RRF_K = 60  # Standard RRF constant

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vectorstore: VectorStore,
        bm25_index: BM25Index,
        chunk_store: ChunkStore,
        settings: Settings,
    ):
        self.embeddings = embedding_service
        self.vectorstore = vectorstore
        self.bm25 = bm25_index
        self.chunk_store = chunk_store
        self.top_k = settings.top_k
        self.bm25_top_k = settings.bm25_top_k
        self.alpha = settings.hybrid_alpha
        self.score_threshold = settings.score_threshold

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_type_filter: str | None = None,
    ) -> list[dict]:
        """Hybrid retrieval: vector search + BM25, fused with RRF."""
        final_k = top_k or self.top_k

        # Path 1: Semantic search via Vertex AI Vector Search
        query_embedding = self.embeddings.embed(query, task_type="RETRIEVAL_QUERY")
        vector_results = self.vectorstore.search(
            query_vector=query_embedding,
            top_k=final_k * 2,
            source_type_filter=source_type_filter,
        )

        # Enrich vector results with text/metadata from chunk store
        for result in vector_results:
            chunk_data = self.chunk_store.get_chunk(result["id"])
            if chunk_data:
                result.update(chunk_data)

        # Path 2: BM25 keyword search
        bm25_results = self.bm25.search(
            query=query,
            top_k=self.bm25_top_k,
            source_type_filter=source_type_filter,
        )

        # Fuse results with Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(vector_results, bm25_results)

        return fused[:final_k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
    ) -> list[dict]:
        """Combine results using RRF scoring."""
        rrf_scores: dict[str, float] = {}
        result_map: dict[str, dict] = {}

        # Score vector results (weighted by alpha)
        for rank, result in enumerate(vector_results):
            doc_id = result["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + self.alpha * (
                1.0 / (self.RRF_K + rank + 1)
            )
            result_map[doc_id] = result

        # Score BM25 results (weighted by 1-alpha)
        for rank, result in enumerate(bm25_results):
            doc_id = result["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 - self.alpha) * (
                1.0 / (self.RRF_K + rank + 1)
            )
            if doc_id not in result_map:
                result_map[doc_id] = result

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for doc_id in sorted_ids:
            entry = result_map[doc_id].copy()
            entry["rrf_score"] = rrf_scores[doc_id]
            results.append(entry)

        return results
