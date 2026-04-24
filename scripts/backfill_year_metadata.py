"""Backfill the `year` field into existing chunk_store entries and BM25 index.

Reads each post's data.json, maps post_id -> year, and updates chunks.json in place.
Does NOT re-embed or touch the vector store — metadata is only held in the chunk store
and BM25 index, both of which are local.
"""

import json
from pathlib import Path

from src.bm25_index import BM25Index
from src.chunk_store import ChunkStore
from src.models import DocumentChunk

DOC_DIR = Path("./data/documents")
CHUNK_STORE_DIR = "./data/chunk_store"
BM25_INDEX_DIR = "./data/bm25_index"


def build_post_year_map(doc_dir: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for post_dir in sorted(doc_dir.iterdir()):
        if not post_dir.is_dir():
            continue
        data_json = post_dir / "data.json"
        if not data_json.exists():
            continue
        with open(data_json, encoding="utf-8") as f:
            meta = json.load(f)
        post_id = meta.get("id")
        year = meta.get("year")
        if post_id is not None and year is not None:
            mapping[int(post_id)] = int(year)
    return mapping


def main() -> None:
    post_year = build_post_year_map(DOC_DIR)
    print(f"Loaded year for {len(post_year)} post(s)")

    chunk_store = ChunkStore(store_dir=CHUNK_STORE_DIR)
    if not chunk_store.load():
        print("No chunk_store found; nothing to backfill.")
        return

    updated = 0
    skipped_no_post = 0
    skipped_no_year = 0
    for entry in chunk_store._chunks.values():
        pid = entry.get("post_id")
        if pid is None:
            skipped_no_post += 1
            continue
        year = post_year.get(int(pid))
        if year is None:
            skipped_no_year += 1
            continue
        if entry.get("year") == year:
            continue
        entry["year"] = year
        updated += 1

    chunk_store._persist()
    print(f"chunks.json updated: {updated} entries (skipped no_post={skipped_no_post}, no_year={skipped_no_year})")

    print("Rebuilding BM25 index with year metadata...")
    bm25 = BM25Index(index_dir=BM25_INDEX_DIR)
    chunks = [
        DocumentChunk(
            text=c["text"],
            source_file=c["source_file"],
            source_type=c["source_type"],
            chunk_index=c["chunk_index"],
            page_number=c.get("page_number"),
            section_title=c.get("section_title"),
            metadata={
                "post_id": c.get("post_id"),
                "post_title": c.get("post_title"),
                "year": c.get("year"),
            },
        )
        for c in chunk_store._chunks.values()
    ]
    bm25.build(chunks)
    bm25.save()
    print(f"BM25 rebuilt: {len(chunks)} chunks")


if __name__ == "__main__":
    main()
