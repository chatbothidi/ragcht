import hashlib
from functools import lru_cache

import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from src.config import Settings


class EmbeddingService:
    def __init__(self, settings: Settings):
        vertexai.init(
            project=settings.gcp_project_id,
            location=settings.gcp_location,
        )
        self.model = TextEmbeddingModel.from_pretrained(settings.embedding_model)
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
        cache_key = hashlib.md5(f"{task_type}:{text}".encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        inputs = [TextEmbeddingInput(text=text, task_type=task_type)]
        result = self.model.get_embeddings(inputs)
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
            results = self.model.get_embeddings(inputs)
            all_embeddings.extend([r.values for r in results])

        return all_embeddings
