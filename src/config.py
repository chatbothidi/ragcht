from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # GCP
    gcp_project_id: str
    gcp_location: str = "asia-northeast3"

    # Vertex AI Models
    embedding_model: str = "text-multilingual-embedding-002"
    llm_model: str = "gemini-2.0-flash"
    llm_location: str | None = None

    # Vector Search — Firestore Vector Search (serverless)
    firestore_database_id: str = "(default)"
    firestore_collection_name: str = "medical_event_chunks"

    # Legacy Vertex AI Vector Search (Matching Engine) — kept for backward compat / rollback
    vertex_collection_name: str = "medical_event_docs"
    vertex_index_id: str | None = None
    vertex_endpoint_id: str | None = None

    # RAG Settings
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 7
    bm25_top_k: int = 10
    hybrid_alpha: float = 0.6
    score_threshold: float = 0.65
    max_context_tokens: int = 4000

    # Reranking
    rerank_candidates: int = 50
    rerank_top_k: int = 10

    # Conversation
    max_conversation_turns: int = 5

    # Redis (required — used for session memory and embedding cache)
    redis_url: str | None = None

    # Embedding cache TTL in seconds (7 days default)
    embedding_cache_ttl: int = 7 * 24 * 3600

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
