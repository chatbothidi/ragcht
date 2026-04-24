import asyncio
import logging
import re

from google.cloud import discoveryengine_v1 as discoveryengine

from src.bm25_index import BM25Index
from src.chunk_store import ChunkStore
from src.config import Settings
from src.embeddings import EmbeddingService
from src.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    RRF_K = 60  # Standard RRF constant
    _RECENCY_PATTERN = re.compile(r"(최근|최신|요즘|올해|금년|이번\s*해)")
    _SORT_PATTERN = re.compile(
        r"(정렬|내림차순|오름차순|날짜순|일자순|순서대로|순서로|나열|리스트|목록)"
    )
    _RECENCY_ALPHA = 0.7  # weight of year vs semantic score when recency intent detected

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

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_type_filter: str | None = None,
    ) -> list[dict]:
        """2-stage retrieval: hybrid search → rerank."""
        final_k = top_k or self.rerank_top_k
        is_recency = bool(self._RECENCY_PATTERN.search(query))
        is_sort = bool(self._SORT_PATTERN.search(query))

        candidates = await self._hybrid_search(query, source_type_filter)

        if not candidates:
            return []

        reranked = await self._rerank(
            query, candidates, top_n=final_k, recency_boost=is_recency or is_sort
        )

        expanded = self._expand_by_post(reranked)

        if is_sort:
            expanded = self._prepare_for_sort(expanded)

        return expanded

    @staticmethod
    def _prepare_for_sort(docs: list[dict]) -> list[dict]:
        """For sort/list intent: dedupe by post_id and present newest-year-first.

        Keeps only the highest-scored chunk per post so the LLM sees one row per
        event. Sorting by year DESC (stable) surfaces latest posts at the top of
        the context; within-year ordering falls back to rerank score.
        """
        seen: dict[str, dict] = {}
        orphans: list[dict] = []
        for d in docs:
            pid = d.get("post_id")
            if pid is None:
                orphans.append(d)
                continue
            key = str(pid)
            score = d.get("rerank_score", d.get("rrf_score", 0)) or 0
            best = seen.get(key)
            best_score = (best.get("rerank_score", best.get("rrf_score", 0)) or 0) if best else -1
            if best is None or score > best_score:
                seen[key] = d

        deduped = list(seen.values()) + orphans
        deduped.sort(
            key=lambda d: (
                d.get("year") if d.get("year") is not None else -1,
                d.get("rerank_score", d.get("rrf_score", 0)) or 0,
            ),
            reverse=True,
        )
        return deduped

    async def _hybrid_search(
        self,
        query: str,
        source_type_filter: str | None = None,
    ) -> list[dict]:
        """Stage 1: Get broad candidates via vector + BM25 + RRF (embed & BM25 in parallel)."""
        query_embedding, bm25_results = await asyncio.gather(
            self.embeddings.embed(query, task_type="RETRIEVAL_QUERY"),
            asyncio.to_thread(
                self.bm25.search,
                query=query,
                top_k=self.rerank_candidates,
                source_type_filter=source_type_filter,
            ),
        )

        vector_results = await self.vectorstore.search(
            query_vector=query_embedding,
            top_k=self.rerank_candidates,
            source_type_filter=source_type_filter,
        )

        for result in vector_results:
            chunk_data = self.chunk_store.get_chunk(result["id"])
            if chunk_data:
                result.update(chunk_data)

        fused = self._reciprocal_rank_fusion(vector_results, bm25_results)

        return fused[: self.rerank_candidates]

    async def _rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int = 10,
        recency_boost: bool = False,
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

        # When recency intent is detected, ask reranker for a larger pool so
        # recency-boosted reordering can surface newer docs into top_n.
        rerank_top_n = min(top_n * 3, len(records)) if recency_boost else top_n

        try:
            request = discoveryengine.RankRequest(
                ranking_config=self._ranking_config,
                model="semantic-ranker-default@latest",
                top_n=rerank_top_n,
                query=query,
                records=records,
            )

            response = await asyncio.to_thread(self._rank_client.rank, request=request)

            reranked = []
            for record in response.records:
                idx = int(record.id)
                entry = candidates[idx].copy()
                entry["rerank_score"] = record.score
                reranked.append(entry)

        except Exception as e:
            logger.warning("[Rerank] Error: %s, falling back to RRF order", e)
            reranked = candidates[:rerank_top_n]

        if recency_boost:
            years = [d.get("year") for d in reranked if d.get("year") is not None]
            if years:
                y_min, y_max = min(years), max(years)
                y_span = max(y_max - y_min, 1)
                scores = [d.get("rerank_score", 0) for d in reranked]
                s_min, s_max = min(scores), max(scores)
                s_span = max(s_max - s_min, 1e-6)
                alpha = self._RECENCY_ALPHA
                for d in reranked:
                    y = d.get("year")
                    s_norm = (d.get("rerank_score", 0) - s_min) / s_span
                    y_norm = ((y - y_min) / y_span) if y is not None else 0.0
                    d["rerank_score"] = (1 - alpha) * s_norm + alpha * y_norm
                reranked.sort(key=lambda d: d.get("rerank_score", 0), reverse=True)
                reranked = reranked[:top_n]

        return reranked

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