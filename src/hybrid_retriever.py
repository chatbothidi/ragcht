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

# 질의에서 연/월 의도 추출 (한국어 "년", "월" suffix 명시 매칭).
_TIME_INTENT_YEAR = re.compile(r"(20\d{2})\s*년")
_TIME_INTENT_MONTH = re.compile(r"(?<!\d)(\d{1,2})\s*월")


def _extract_time_intent(query: str) -> dict:
    """Query에서 year(int)와 months(list[int]) 추출. 없으면 None/[].

    "2026년 3월" → {year: 2026, months: [3]}
    "2025년" → {year: 2025, months: []}
    "1월부터 3월까지" → {year: None, months: [1, 3]}
    """
    year_match = _TIME_INTENT_YEAR.search(query)
    year = int(year_match.group(1)) if year_match else None
    months = [int(m) for m in _TIME_INTENT_MONTH.findall(query) if 1 <= int(m) <= 12]
    return {"year": year, "months": months}

class HybridRetriever:
    RRF_K = 60  # 표준 RRF 상수
    _RECENCY_PATTERN = re.compile(r"(최근|최신|요즘|올해|금년|이번\s*해)")
    _SORT_PATTERN = re.compile(
        r"(정렬|내림차순|오름차순|날짜순|일자순|순서대로|순서로|나열|리스트|목록)"
    )
    _RECENCY_ALPHA = 0.7  # 최신 의도가 감지됐을 때 연도 vs 의미 점수의 가중치

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

        # Discovery Engine Ranking 클라이언트 (GCP)
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
        """2단계 검색: 하이브리드 검색 → 재순위(rerank)."""
        final_k = top_k or self.rerank_top_k
        is_recency = bool(self._RECENCY_PATTERN.search(query))
        is_sort = bool(self._SORT_PATTERN.search(query))

        candidates = await self._hybrid_search(query, source_type_filter)

        if not candidates:
            return []

        reranked = await self._rerank(
            query, candidates, top_n=final_k, recency_boost=is_recency or is_sort
        )

        # 시간 의도 부스트 — query에 명시적 year/month가 있으면 일치 청크 점수 ↑
        time_intent = _extract_time_intent(query)
        if time_intent["year"] or time_intent["months"]:
            self._apply_time_boost(reranked, time_intent)

        reranked = [d for d in reranked if d.get("rerank_score", 0) >= self.score_threshold]
        if not reranked:
            return []

        expanded = self._expand_by_post(reranked)

        if is_sort:
            expanded = self._prepare_for_sort(expanded)

        return expanded

    @staticmethod
    def _prepare_for_sort(docs: list[dict]) -> list[dict]:
        """정렬/목록 의도용: post_id 기준 중복 제거 후 최신 연도부터 정렬.

        게시물당 가장 점수가 높은 청크만 유지해 LLM이 이벤트당 한 행만 보도록 함.
        연도 내림차순(안정 정렬)으로 최신 게시물을 컨텍스트 상단에 배치하며,
        같은 연도 내에서는 rerank 점수 순으로 정렬.
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
        """1단계: 벡터 + BM25 + RRF로 광범위한 후보 수집 (임베딩과 BM25 병렬 실행).

        질의에 명시 year(예: "2026년")이 있으면 vector/BM25 검색 모두에 year 필터 적용.
        후보 풀이 해당 연도 docs로 좁아져 짧은 doc도 안 밀려남.
        """
        time_intent = _extract_time_intent(query)
        year_filter = time_intent.get("year")

        query_embedding, bm25_results = await asyncio.gather(
            self.embeddings.embed(query, task_type="RETRIEVAL_QUERY"),
            asyncio.to_thread(
                self.bm25.search,
                query=query,
                top_k=self.rerank_candidates,
                source_type_filter=source_type_filter,
                year_filter=year_filter,
            ),
        )

        vector_results = await self.vectorstore.search(
            query_vector=query_embedding,
            top_k=self.rerank_candidates,
            source_type_filter=source_type_filter,
            year_filter=year_filter,
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
        """2단계: Discovery Engine Ranking API로 후보 재순위."""

        # 랭킹 레코드 생성
        records = []
        for i, doc in enumerate(candidates):
            text = doc.get("text", "")
            title = doc.get("post_title") or doc.get("source_file", "")
            records.append(
                discoveryengine.RankingRecord(
                    id=str(i),
                    title=title,
                    content=text[:1000],
                )
            )

        if not records:
            return candidates[:top_n]

        # 새로운 내용이 추가됬을때 검색풀을 늘려서 다시 재정립하기 (새로운 내용이 나왔는데 기존에 검색했던 것을 토대로 하면 검색이 할루시네이션이 발생하기 때문)
        # (-> 이전 conversation turn이 5개여서 그 내에 있는 것들을 기준으로 우선 검색을 해서 검색 시간을 줄여왔음)
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
            reranked = []
            for entry in candidates[:rerank_top_n]:
                copy = entry.copy()
                copy["rerank_score"] = 1.0
                copy["rerank_fallback"] = True
                reranked.append(copy)

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

    def _apply_time_boost(self, reranked: list[dict], time_intent: dict) -> None:
        """청크의 year 또는 텍스트의 월이 query 의도와 일치하면 rerank_score 부스트.

        부스트 후 점수 내림차순으로 reranked 리스트 in-place 재정렬.
        - year 일치: score *= 2.5
        - text에 target month 등장: score *= 1.8
        - 둘 다 일치: 누적 곱 (2.5 * 1.8 = 4.5)
        """
        YEAR_BOOST = 2.5
        MONTH_BOOST = 1.8
        target_year = time_intent.get("year")
        target_months = time_intent.get("months", [])
        # 월 텍스트 매칭: 앞 자리에 다른 숫자가 없어야 함 (예: "12월" 안의 "2월" 오매칭 방지)
        month_patterns = [re.compile(rf"(?<!\d){m}월") for m in target_months]

        for d in reranked:
            score = d.get("rerank_score", 0)
            if target_year and d.get("year") == target_year:
                score *= YEAR_BOOST
            if target_months:
                text = d.get("text", "")
                if any(p.search(text) for p in month_patterns):
                    score *= MONTH_BOOST
            d["rerank_score"] = score
        reranked.sort(key=lambda d: d.get("rerank_score", 0), reverse=True)

    def _expand_by_post(
        self,
        results: list[dict],
        max_chunks_per_post: int = 5,
    ) -> list[dict]:
        """3단계: 각 결과에 대해 같은 게시물의 다른 청크 포함."""
        if not results:
            return results

        # 검색결과의 게시물 ID 모으기
        seen_post_ids = set()
        seen_chunk_ids = set()
        for r in results:
            pid = r.get("post_id")
            if pid is not None:
                seen_post_ids.add(str(pid))
            seen_chunk_ids.add(r.get("id"))

        if not seen_post_ids:
            return results

        # 같은 게시물에 있는 필요한 chunk들 추가
        additional = []
        for cid, chunk_data in self.chunk_store._chunks.items():
            if cid in seen_chunk_ids:
                continue
            chunk_pid = str(chunk_data.get("post_id", ""))
            if chunk_pid in seen_post_ids:
                additional.append({
                    "id": cid,
                    **chunk_data,
                    "rerank_score": 0,  
                    "expanded": True,
                })

        # 청크들의 앞 뒤 맥락을 파악을 위해 여러개 검색
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
    ) -> list[dict]: # RRF 알고리즘 (bm25와 vector search를 등수 기반으로 재정립하는 로직)
        """RRF 스코어링으로 결과 결합."""
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