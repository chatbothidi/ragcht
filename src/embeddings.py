import hashlib
import random
import time
from functools import lru_cache

import vertexai
from google.api_core.exceptions import (
    DeadlineExceeded,
    InternalServerError,
    ResourceExhausted,
    ServiceUnavailable,
)
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from src.config import Settings

_RETRYABLE_EXCEPTIONS = (
    ServiceUnavailable,
    DeadlineExceeded,
    InternalServerError,
    ResourceExhausted,
)


class EmbeddingService:
    def __init__(self, settings: Settings):
        vertexai.init(
            project=settings.gcp_project_id,
            location=settings.gcp_location,
        )
        self.model = TextEmbeddingModel.from_pretrained(settings.embedding_model)
        self._cache: dict[str, list[float]] = {}

    def _get_embeddings_with_retry(
        self,
        inputs: list[TextEmbeddingInput],
        max_attempts: int = 6,
        initial_delay: float = 2.0,
    ):
        """Call Vertex AI embedding API with exponential backoff on transient errors."""
        delay = initial_delay
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return self.model.get_embeddings(inputs)
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt == max_attempts:
                    break
                sleep_for = delay + random.uniform(0, delay * 0.25)
                print(
                    f"  [embed] {type(exc).__name__} on attempt {attempt}/{max_attempts}; "
                    f"retrying in {sleep_for:.1f}s"
                )
                time.sleep(sleep_for)
                delay *= 2
        raise last_exc  # type: ignore[misc]

    def embed(self, text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
        cache_key = hashlib.md5(f"{task_type}:{text}".encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        inputs = [TextEmbeddingInput(text=text, task_type=task_type)]
        result = self._get_embeddings_with_retry(inputs)
        embedding = result[0].values

        self._cache[cache_key] = embedding
        return embedding

    def embed_batch(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 5,
    ) -> list[list[float]]:
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = [TextEmbeddingInput(text=t, task_type=task_type) for t in batch]
            results = self._get_embeddings_with_retry(inputs)
            all_embeddings.extend([r.values for r in results])

        return all_embeddings
