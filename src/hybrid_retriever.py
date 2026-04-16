from google.cloud import discoveryengine_v1 as discoveryengine

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
        self.rerank_candidates = settings.rerank_candidates
        self.rerank_top_k = settings.rerank_top_k

        # Discovery Engine Ranking client
        self._rank_client = discoveryengine.RankServiceClient()
        self._ranking_config = self._rank_client.ranking_config_path(
            project=settings.gcp_project_id,
            location="global",
            ranking_config="default_ranking_config",
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_type_filter: str | None = None,
    ) -> list[dict]:
        """2-stage retrieval: hybrid search → rerank."""
        final_k = top_k or self.rerank_top_k

        # Stage 1: Broad hybrid search
        candidates = self._hybrid_search(query, source_type_filter)

        if not candidates:
            return []

        # Stage 2: Rerank with Discovery Engine
        reranked = self._rerank(query, candidates, top_n=final_k)

        # Stage 3: Expand by post_id - include other chunks from same posts
        expanded = self._expand_by_post(reranked)

        return expanded

    def _hybrid_search(
        self,
        query: str,
        source_type_filter: str | None = None,
    ) -> list[dict]:
        """Stage 1: Get broad candidates via vector + BM25 + RRF."""
        # Vector Search - wide net
        query_embedding = self.embeddings.embed(query, task_type="RETRIEVAL_QUERY")
        vector_results = self.vectorstore.search(
            query_vector=query_embedding,
            top_k=self.rerank_candidates,
            source_type_filter=source_type_filter,
        )

        # Enrich with text/metadata from chunk store
        for result in vector_results:
            chunk_data = self.chunk_store.get_chunk(result["id"])
            if chunk_data:
                result.update(chunk_data)

        # BM25 search - wide net
        bm25_results = self.bm25.search(
            query=query,
            top_k=self.rerank_candidates,
            source_type_filter=source_type_filter,
        )

        # RRF fusion
        fused = self._reciprocal_rank_fusion(vector_results, bm25_results)

        return fused[:self.rerank_candidates]

    def _rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int = 10,
    ) -> list[dict]:
        """Stage 2: Rerank candidates using Discovery Engine Ranking API."""
        # Build ranking records
        records = []
        for i, doc in enumerate(candidates):
            text = doc.get("text", "")
            title = doc.get("post_title") or doc.get("source_file", "")
            records.append(
                discoveryengine.RankingRecord(
                    id=str(i),
                    title=title,
                    content=text[:1000],  # Limit content length
                )
            )

        if not records:
            return candidates[:top_n]

        try:
            request = discoveryengine.RankRequest(
                ranking_config=self._ranking_config,
                model="semantic-ranker-default@latest",
                top_n=top_n,
                query=query,
                records=records,
            )

            response = self._rank_client.rank(request=request)

            # Map back to original candidates with rerank scores
            reranked = []
            for record in response.records:
                idx = int(record.id)
                entry = candidates[idx].copy()
                entry["rerank_score"] = record.score
                reranked.append(entry)

            return reranked

        except Exception as e:
            print(f"[Rerank] Error: {e}, falling back to RRF order")
            return candidates[:top_n]

    def _expand_by_post(
        self,
        results: list[dict],
        max_chunks_per_post: int = 5,
    ) -> list[dict]:
        """Stage 3: For each result, include other chunks from the same post."""
        if not results:
            return results

        # Collect post_ids from search results
        seen_post_ids = set()
        seen_chunk_ids = set()
        for r in results:
            pid = r.get("post_id")
            if pid is not None:
                seen_post_ids.add(str(pid))
            seen_chunk_ids.add(r.get("id"))

        if not seen_post_ids:
            return results

        # Find additional chunks from same posts in chunk_store
        additional = []
        for cid, chunk_data in self.chunk_store._chunks.items():
            if cid in seen_chunk_ids:
                continue
            chunk_pid = str(chunk_data.get("post_id", ""))
            if chunk_pid in seen_post_ids:
                additional.append({
                    "id": cid,
                    **chunk_data,
                    "rerank_score": 0,  # Not reranked, added by expansion
                    "expanded": True,
                })

        # Limit additional chunks per post
        post_counts: dict[str, int] = {}
        limited_additional = []
        for chunk in additional:
            pid = str(chunk.get("post_id", ""))
            post_counts[pid] = post_counts.get(pid, 0) + 1
            if post_counts[pid] <= max_chunks_per_post:
                limited_additional.append(chunk)

        return results + limited_additional

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
    ) -> list[dict]:
        """Combine results using RRF scoring."""
        rrf_scores: dict[str, float] = {}
        result_map: dict[str, dict] = {}

        for rank, result in enumerate(vector_results):
            doc_id = result["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + self.alpha * (
                1.0 / (self.RRF_K + rank + 1)
            )
            result_map[doc_id] = result

        for rank, result in enumerate(bm25_results):
            doc_id = result["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 - self.alpha) * (
                1.0 / (self.RRF_K + rank + 1)
            )
            if doc_id not in result_map:
                result_map[doc_id] = result

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for doc_id in sorted_ids:
            entry = result_map[doc_id].copy()
            entry["rrf_score"] = rrf_scores[doc_id]
            results.append(entry)

        return results
