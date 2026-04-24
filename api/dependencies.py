from functools import lru_cache

from redis.asyncio import Redis

from src.bm25_index import BM25Index
from src.chunk_store import ChunkStore
from src.config import Settings, get_settings
from src.embeddings import EmbeddingService
from src.generator import LLMGenerator
from src.hybrid_retriever import HybridRetriever
from src.memory import ConversationMemory
from src.rag_pipeline import RAGPipeline
from src.vectorstore import VectorStore


@lru_cache
def get_redis() -> Redis:
    settings = get_settings()
    if not settings.redis_url:
        raise RuntimeError(
            "REDIS_URL is required. Set it in .env or via environment variable."
        )
    return Redis.from_url(settings.redis_url, decode_responses=True)


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(get_settings(), redis_client=get_redis())


@lru_cache
def get_vectorstore() -> VectorStore:
    return VectorStore(get_settings())


@lru_cache
def get_bm25_index() -> BM25Index:
    index = BM25Index()
    index.load()
    return index


@lru_cache
def get_generator() -> LLMGenerator:
    return LLMGenerator(get_settings())


@lru_cache
def get_memory() -> ConversationMemory:
    settings = get_settings()
    return ConversationMemory(
        redis_client=get_redis(),
        max_turns=settings.max_conversation_turns,
    )


@lru_cache
def get_chunk_store() -> ChunkStore:
    store = ChunkStore()
    store.load()
    return store


@lru_cache
def get_retriever() -> HybridRetriever:
    return HybridRetriever(
        embedding_service=get_embedding_service(),
        vectorstore=get_vectorstore(),
        bm25_index=get_bm25_index(),
        chunk_store=get_chunk_store(),
        settings=get_settings(),
    )


@lru_cache
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(
        retriever=get_retriever(),
        generator=get_generator(),
        memory=get_memory(),
        settings=get_settings(),
    )
