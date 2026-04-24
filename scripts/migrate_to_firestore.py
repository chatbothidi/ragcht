#!/usr/bin/env python3
"""One-shot migration: read chunks.json → re-embed → upsert to Firestore.

Bypasses DocumentLoader / OCR / Chunker entirely. Uses the already-generated
chunk text from `data/chunk_store/chunks.json`. This is the fast path for
migrating to Firestore without re-processing PDFs/images.

Usage:
    python scripts/migrate_to_firestore.py          # full
    python scripts/migrate_to_firestore.py --limit 100  # test with 100 chunks
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.embeddings import EmbeddingService
from src.models import DocumentChunk as DC
from src.vectorstore import VectorStore
from api.dependencies import get_redis


def _load_chunks(limit: int | None = None) -> list[tuple[str, dict]]:
    path = Path("data/chunk_store/chunks.json")
    with path.open() as f:
        store = json.load(f)
    items = list(store.items())
    if limit:
        items = items[:limit]
    return items


def _dict_to_chunk(d: dict) -> DC:
    return DC(
        text=d["text"],
        source_file=d["source_file"],
        source_type=d["source_type"],
        chunk_index=d["chunk_index"],
        page_number=d.get("page_number"),
        section_title=d.get("section_title"),
        metadata={
            "post_id": d.get("post_id"),
            "post_title": d.get("post_title"),
            "year": d.get("year"),
            "attachments": d.get("attachments") or [],
        },
    )


async def main_async(args):
    settings = get_settings()
    redis = get_redis()
    embeddings = EmbeddingService(settings, redis_client=redis)
    vectorstore = VectorStore(settings)

    try:
        print("Loading chunks.json...")
        items = _load_chunks(limit=args.limit)
        print(f"  {len(items)} chunks to migrate")

        if args.clear:
            print("Clearing existing Firestore collection...")
            to_delete = []
            async for snap in vectorstore._collection().stream():
                to_delete.append(snap.id)
            if to_delete:
                await vectorstore.remove(to_delete)
            print(f"  Removed {len(to_delete)} existing docs")

        # Process in page-sized batches to show progress + avoid holding everything at once
        PAGE = 500
        total = len(items)
        upserted = 0
        for start in range(0, total, PAGE):
            page_items = items[start : start + PAGE]
            chunks = [_dict_to_chunk(d) for _, d in page_items]
            texts = [c.text for c in chunks]

            print(f"Page {start}-{start + len(page_items)}: embedding {len(texts)}...")
            vecs = await embeddings.embed_batch(texts)
            print(f"  embedded {len(vecs)}")

            print(f"  upserting to Firestore...")
            ids = await vectorstore.upsert(chunks, vecs)
            upserted += len(ids)
            print(f"  cumulative upserted: {upserted}/{total}")

        print(f"\nMigration complete: {upserted} docs in Firestore")
    finally:
        await vectorstore.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only first N chunks (testing)")
    parser.add_argument("--clear", action="store_true", help="Delete existing Firestore docs first")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
