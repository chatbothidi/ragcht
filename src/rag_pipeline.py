from collections.abc import Generator
from typing import Any

from src.config import Settings
from src.generator import LLMGenerator
from src.hybrid_retriever import HybridRetriever
from src.memory import ConversationMemory
from src.models import RAGResponse, SourceCitation


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

    def query(
        self,
        question: str,
        session_id: str | None = None,
        source_type_filter: str | None = None,
    ) -> RAGResponse:
        """Execute full RAG pipeline."""
        # Session management
        if not session_id:
            session_id = self.memory.create_session()

        # Load conversation history
        history = self.memory.get_history(session_id)

        # Query rewriting for multi-turn conversations
        rewritten_query = None
        search_query = question
        if history:
            rewritten_query = self.generator.rewrite_query(question, history)
            search_query = rewritten_query

        # Retrieve
        documents = self.retriever.retrieve(
            query=search_query,
            source_type_filter=source_type_filter,
        )

        if not documents:
            answer = "제공된 문서에서 관련 정보를 찾을 수 없습니다."
            self.memory.add_turn(session_id, question, answer)
            return RAGResponse(
                answer=answer,
                sources=[],
                query=question,
                rewritten_query=rewritten_query,
                session_id=session_id,
            )

        # Build context and generate
        context = self.generator.build_context(documents)
        answer = self.generator.generate(
            query=search_query,
            context=context,
            conversation_history=history,
        )

        # Save to memory
        self.memory.add_turn(session_id, question, answer)

        # Build source citations
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

    def query_stream(
        self,
        question: str,
        session_id: str | None = None,
        source_type_filter: str | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Execute RAG pipeline with streaming response."""
        if not session_id:
            session_id = self.memory.create_session()

        history = self.memory.get_history(session_id)

        rewritten_query = None
        search_query = question
        if history:
            rewritten_query = self.generator.rewrite_query(question, history)
            search_query = rewritten_query

        documents = self.retriever.retrieve(
            query=search_query,
            source_type_filter=source_type_filter,
        )

        if not documents:
            yield {"type": "token", "content": "제공된 문서에서 관련 정보를 찾을 수 없습니다."}
            yield {"type": "done", "session_id": session_id}
            return

        context = self.generator.build_context(documents)

        # Stream tokens
        full_answer = []
        for token in self.generator.generate_stream(
            query=search_query,
            context=context,
            conversation_history=history,
        ):
            full_answer.append(token)
            yield {"type": "token", "content": token}

        # Save to memory
        self.memory.add_turn(session_id, question, "".join(full_answer))

        # Yield sources
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
