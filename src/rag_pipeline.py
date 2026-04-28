import logging
import re
from collections.abc import AsyncGenerator
from datetime import date
from typing import Any

from google.genai import errors as genai_errors

from src.config import Settings
from src.generator import LLMGenerator
from src.hybrid_retriever import HybridRetriever
from src.memory import ConversationMemory
from src.models import RAGResponse, SourceCitation

logger = logging.getLogger(__name__)

QUOTA_FALLBACK_MESSAGE = (
    "현재 요청이 몰려 답변을 생성할 수 없습니다. 잠시 후 다시 시도해 주세요."
)
GENERIC_FALLBACK_MESSAGE = (
    "답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
)

def _fallback_message(exc: Exception) -> str:
    if isinstance(exc, genai_errors.ClientError) and getattr(exc, "code", None) == 429:
        return QUOTA_FALLBACK_MESSAGE
    return GENERIC_FALLBACK_MESSAGE


def _normalize_relative_years(text: str) -> str:
    """Replace Korean relative-year terms with absolute years based on today.

    Longer terms first to avoid corrupting overlapping shorter terms
    (예: "재작년" contains "작년").
    """
    current_year = date.today().year
    replacements = [
        ("재작년", current_year - 2),
        ("내후년", current_year + 2),
        ("지난해", current_year - 1),
        ("올해", current_year),
        ("금년", current_year),
        ("작년", current_year - 1),
        ("내년", current_year + 1),
        ("명년", current_year + 1),
    ]
    for term, year in replacements:
        text = text.replace(term, f"{year}년")
    return text


_SELF_CONTAINED_YEAR = re.compile(r"\b(19|20|21|22|23|24|25|26)\d{2}\b")
_SELF_CONTAINED_EVENT = re.compile(
    r"(Workshop|workshop|School|school|Symposium|symposium|"
    r"심포지엄|학술대회|연수강좌|워크숍|컨퍼런스|세미나|학회)"
)
_FOLLOWUP_PRONOUN = re.compile(r"(그것|그거|그 행사|그 학회|거기|이번|지난번|방금|아까|위에|이전)")


def _is_self_contained(text: str) -> bool:
    """Return True when the query has explicit topic markers and no follow-up
    pronouns — meaning prior conversation history is more likely to harm than
    help. Used to bypass rewrite_query and clear history for those queries.
    """
    if _FOLLOWUP_PRONOUN.search(text):
        return False
    has_year = bool(_SELF_CONTAINED_YEAR.search(text))
    has_event = bool(_SELF_CONTAINED_EVENT.search(text))
    return has_year or has_event


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: LLMGenerator,
        memory: ConversationMemory,
        settings: Settings,
    ):
        self.retriever = retriever
        self.generator = generator
        self.memory = memory
        self.top_k = settings.top_k

    async def query(
        self,
        question: str,
        session_id: str | None = None,
        source_type_filter: str | None = None,
    ) -> RAGResponse:
        """Full RAG pipeline."""
        if not session_id:
            session_id = self.memory.create_session()

        history = await self.memory.get_history(session_id)

        normalized_question = _normalize_relative_years(question)
        self_contained = _is_self_contained(normalized_question)
        effective_history = None if self_contained else history

        rewritten_query = None
        search_query = normalized_question
        if effective_history:
            rewritten_query = await self.generator.rewrite_query(normalized_question, effective_history)
            search_query = rewritten_query

        documents = await self.retriever.retrieve(
            query=search_query,
            source_type_filter=source_type_filter,
        )

        if not documents:
            answer = "제공된 문서에서 관련 정보를 찾을 수 없습니다."
            await self.memory.add_turn(session_id, question, answer)
            return RAGResponse(
                answer=answer,
                sources=[],
                query=question,
                rewritten_query=rewritten_query,
                session_id=session_id,
            )

        context = self.generator.build_context(documents)
        answer = await self.generator.generate(
            query=search_query,
            context=context,
            conversation_history=effective_history,
        )

        await self.memory.add_turn(session_id, question, answer)

        sources = [
            SourceCitation(
                source_file=doc.get("source_file", "Unknown"),
                source_type=doc.get("source_type", "unknown"),
                page_number=doc.get("page_number"),
                relevance_score=round(doc.get("rrf_score", doc.get("score", 0)), 4),
                excerpt=doc.get("text", "")[:150] + "...",
            )
            for doc in documents
        ]

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            rewritten_query=rewritten_query,
            session_id=session_id,
        )

    async def query_stream(
        self,
        question: str,
        session_id: str | None = None,
        source_type_filter: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Execute RAG pipeline with streaming response."""
        if not session_id:
            session_id = self.memory.create_session()

        try:
            history = await self.memory.get_history(session_id)

            normalized_question = _normalize_relative_years(question)
            self_contained = _is_self_contained(normalized_question)
            effective_history = None if self_contained else history

            rewritten_query = None
            search_query = normalized_question
            if effective_history:
                rewritten_query = await self.generator.rewrite_query(normalized_question, effective_history)
                search_query = rewritten_query

            documents = await self.retriever.retrieve(
                query=search_query,
                source_type_filter=source_type_filter,
            )

        except Exception as exc:
            logger.exception("query_stream setup failed")
            msg = _fallback_message(exc)
            yield {"type": "token", "content": msg}
            yield {"type": "done", "session_id": session_id}
            return

        if not documents:
            logger.warning(
                "[query_stream] NO documents retrieved | query=%r rewritten=%r filter=%r session=%s",
                question[:80], rewritten_query, source_type_filter, session_id,
            )
            yield {"type": "token", "content": "제공된 문서에서 관련 정보를 찾을 수 없습니다."}
            yield {"type": "done", "session_id": session_id}
            return

        context = self.generator.build_context(documents)

        full_answer: list[str] = []
        stream_failed: Exception | None = None
        try:
            async for token in self.generator.generate_stream(
                query=search_query,
                context=context,
                conversation_history=effective_history,
            ):
                full_answer.append(token)
                yield {"type": "token", "content": token}
        except Exception as exc:
            logger.exception("generate_stream failed")
            stream_failed = exc

        if stream_failed is not None or not full_answer:
            msg = _fallback_message(stream_failed) if stream_failed else GENERIC_FALLBACK_MESSAGE
            if not full_answer:
                yield {"type": "token", "content": msg}
            yield {"type": "done", "session_id": session_id}
            return

        await self.memory.add_turn(session_id, question, "".join(full_answer))

        sources = [
            {
                "source_file": doc.get("source_file", "Unknown"),
                "source_type": doc.get("source_type", "unknown"),
                "page_number": doc.get("page_number"),
                "relevance_score": round(doc.get("rrf_score", doc.get("score", 0)), 4),
            }
            for doc in documents
        ]
        yield {"type": "sources", "data": sources}
        yield {"type": "done", "session_id": session_id}