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
            entry = {
                "text": chunk.text,
                "source_file": chunk.source_file,
                "source_type": chunk.source_type,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
            }
            # 게시물 메타데이터 추가
            if chunk.metadata:
                if "post_id" in chunk.metadata:
                    entry["post_id"] = chunk.metadata["post_id"]
                if "post_title" in chunk.metadata:
                    entry["post_title"] = chunk.metadata["post_title"]
                if "attachments" in chunk.metadata:
                    entry["attachments"] = chunk.metadata["attachments"]
                if chunk.metadata.get("year") is not None:
                    entry["year"] = chunk.metadata["year"]
                if chunk.metadata.get("url"):
                    entry["url"] = chunk.metadata["url"]
            self._chunks[chunk_id] = entry
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
        """이미 있는 인덱싱 된 파일을 보여줌"""
        return {c["source_file"] for c in self._chunks.values()}

    def get_ids_by_source(self, source_file: str) -> list[str]:
        """특정 Chunk ID를 가진 파일 보여줌"""
        return [cid for cid, c in self._chunks.items() if c["source_file"] == source_file]

    def remove_by_source(self, source_file: str) -> list[str]:
        """해당 파일에 관련된 Chunks 데이터 모두 지우기. 지워진 ID 보여줌"""
        ids_to_remove = self.get_ids_by_source(source_file)
        for cid in ids_to_remove:
            del self._chunks[cid]
        if ids_to_remove:
            self._persist()
        return ids_to_remove
