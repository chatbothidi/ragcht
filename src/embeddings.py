import asyncio
import hashlib
import json
import logging
import random

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from redis.asyncio import Redis

from src.config import Settings

logger = logging.getLogger(__name__)

_RETRYABLE_EXCEPTIONS = (
    genai_errors.ServerError,
    httpx.TimeoutException,
    httpx.ConnectError,
)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    if isinstance(exc, genai_errors.ClientError) and getattr(exc, "code", None) == 429:
        return True
    return False


class EmbeddingService:
    def __init__(self, settings: Settings, redis_client: Redis | None = None):
        self.client = genai.Client(
            vertexai=True,
            project=settings.gcp_project_id,
            location=settings.gcp_location,
        )
        self.model_name = settings.embedding_model
        self.redis = redis_client
        self.cache_ttl = settings.embedding_cache_ttl

    def _cache_key(self, text: str, task_type: str) -> str:
        digest = hashlib.sha256(
            f"{task_type}:{self.model_name}:{text}".encode()
        ).hexdigest()
        return f"emb:{digest}"

    async def _embed_with_retry(
        self,
        contents: list[str],
        task_type: str,
        max_attempts: int = 3,
        initial_delay: float = 2.0,
    ) -> types.EmbedContentResponse:
        """Call Vertex AI embedding API via google-genai with exponential backoff."""
        delay = initial_delay
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await self.client.aio.models.embed_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
            except Exception as exc:
                if not _is_retryable(exc):
                    raise
                last_exc = exc
                if attempt == max_attempts:
                    break
                is_quota = (
                    isinstance(exc, genai_errors.ClientError)
                    and getattr(exc, "code", None) == 429
                )
                base = max(delay, 8.0) if is_quota else delay
                sleep_for = base + random.uniform(0, base * 0.25)
                logger.warning(
                    "[embed] %s on attempt %d/%d; retrying in %.1fs",
                    type(exc).__name__,
                    attempt,
                    max_attempts,
                    sleep_for,
                )
                await asyncio.sleep(sleep_for)
                delay *= 2
        raise last_exc  # type: ignore[misc]

    async def embed(self, text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
        if self.redis is not None:
            key = self._cache_key(text, task_type)
            cached = await self.redis.get(key)
            if cached:
                logger.debug("embed_cache_hit key=%s", key[:16])
                return json.loads(cached)

        response = await self._embed_with_retry([text], task_type)
        embedding = list(response.embeddings[0].values)

        if self.redis is not None:
            key = self._cache_key(text, task_type)
            await self.redis.setex(key, self.cache_ttl, json.dumps(embedding))
            logger.debug("embed_cache_miss key=%s", key[:16])

        return embedding

    async def embed_batch(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 5,
    ) -> list[list[float]]:
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self._embed_with_retry(batch, task_type)
            all_embeddings.extend([list(e.values) for e in response.embeddings])

        return all_embeddings
