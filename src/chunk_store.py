"""Local chunk store for mapping Vector Search IDs back to chunk text/metadata."""

import json
from pathlib import Path

from src.models import DocumentChunk


class ChunkStore:
    def __init__(self, store_dir: str = "./data/chunk_store"):
        self.store_dir = Path(store_dir)
        self._chunks: dict[str, dict] = {}

    def save_chunks(self, chunk_ids: list[str], chunks: list[DocumentChunk]) -> None:
        """Save chunk ID -> metadata mapping."""
        for chunk_id, chunk in zip(chunk_ids, chunks):
            self._chunks[chunk_id] = {
                "text": chunk.text,
                "source_file": chunk.source_file,
                "source_type": chunk.source_type,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
            }
        self._persist()

    def get_chunk(self, chunk_id: str) -> dict | None:
        return self._chunks.get(chunk_id)

    def get_chunks(self, chunk_ids: list[str]) -> list[dict]:
        return [self._chunks.get(cid, {}) for cid in chunk_ids]

    def _persist(self) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        with open(self.store_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self._chunks, f, ensure_ascii=False)

    def load(self) -> bool:
        path = self.store_dir / "chunks.json"
        if not path.exists():
            return False
        with open(path, encoding="utf-8") as f:
            self._chunks = json.load(f)
        return True

    def get_indexed_files(self) -> set[str]:
        """Return set of source_file names already indexed."""
        return {c["source_file"] for c in self._chunks.values()}

    def get_ids_by_source(self, source_file: str) -> list[str]:
        """Return chunk IDs for a given source file."""
        return [cid for cid, c in self._chunks.items() if c["source_file"] == source_file]

    def remove_by_source(self, source_file: str) -> list[str]:
        """Remove all chunks for a source file. Returns removed IDs."""
        ids_to_remove = self.get_ids_by_source(source_file)
        for cid in ids_to_remove:
            del self._chunks[cid]
        if ids_to_remove:
            self._persist()
        return ids_to_remove
