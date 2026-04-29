#!/usr/bin/env python3
"""Incremental / full document indexing into Firestore Vector Search + BM25.

Usage:
    python scripts/index_documents.py          # Incremental (new/deleted posts only)
    python scripts/index_documents.py --full   # Full re-index (clears Firestore first)
    python scripts/index_documents.py --add 12345
    python scripts/index_documents.py --delete 12345
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from redis.asyncio import Redis

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bm25_index import BM25Index
from src.chunk_store import ChunkStore
from src.chunker import DocumentChunker
from src.config import get_settings
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingService
from src.models import DocumentChunk as DC
from src.vectorstore import VectorStore


def _scan_post_ids(doc_dir: str) -> set[str]:
    """Scan document directory for post IDs (folders with data.json)."""
    base = Path(doc_dir)
    post_ids = set()
    for d in base.iterdir():
        if d.is_dir() and (d / "data.json").exists():
            post_ids.add(d.name)
    return post_ids


def _get_indexed_post_ids(chunk_store: ChunkStore) -> set[str]:
    post_ids = set()
    for c in chunk_store._chunks.values():
        pid = c.get("post_id")
        if pid is not None:
            post_ids.add(str(pid))
    return post_ids


def _rebuild_bm25(chunk_store: ChunkStore, bm25: BM25Index) -> None:
    all_chunks = [
        DC(
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
    bm25.build(all_chunks)
    bm25.save()


async def _clear_firestore(vectorstore: VectorStore) -> int:
    """Delete all documents in the Firestore collection (full re-index only)."""
    coll = vectorstore._collection()
    to_delete: list[str] = []
    async for snap in coll.stream():
        to_delete.append(snap.id)
    if to_delete:
        await vectorstore.remove(to_delete)
    return len(to_delete)


async def full_index(loader, chunker, embeddings, vectorstore, chunk_store, bm25, doc_dir):
    """Full re-index: clear Firestore collection and re-upload everything."""
    print(f"Loading documents from {doc_dir}...")
    documents = loader.load_directory(doc_dir)
    print(f"  Loaded {len(documents)} documents")

    if not documents:
        print("No documents found.")
        return

    medical_count = sum(1 for d in documents if d.source_type == "medical")
    event_count = sum(1 for d in documents if d.source_type == "event")
    print(f"  Medical: {medical_count}, Event: {event_count}")

    print("\nChunking documents...")
    chunks = chunker.chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    print("\nGenerating embeddings (this may take a while)...")
    chunk_texts = [c.text for c in chunks]
    chunk_embeddings = await embeddings.embed_batch(chunk_texts)
    print(f"  Generated {len(chunk_embeddings)} embeddings")

    print("\nClearing existing Firestore documents...")
    removed = await _clear_firestore(vectorstore)
    print(f"  Removed {removed} old documents")

    print("\nIndexing in Firestore Vector Search...")
    chunk_ids = await vectorstore.upsert(chunks, chunk_embeddings)
    print(f"  Firestore indexing complete ({len(chunk_ids)} docs)")

    print("\nSaving chunk store...")
    chunk_store._chunks = {}
    chunk_store.save_chunks(chunk_ids, chunks)
    print("  Chunk store saved")

    print("\nBuilding BM25 index...")
    bm25.build(chunks)
    bm25.save()
    print("  BM25 index saved")

    print(f"\nFull indexing complete! Total chunks: {len(chunks)}")


async def incremental_index(loader, chunker, embeddings, vectorstore, chunk_store, bm25, doc_dir):
    """Incremental index: compare post IDs on disk vs chunk store."""
    chunk_store.load()

    disk_post_ids = _scan_post_ids(doc_dir)
    indexed_post_ids = _get_indexed_post_ids(chunk_store)

    new_posts = disk_post_ids - indexed_post_ids
    deleted_posts = indexed_post_ids - disk_post_ids

    if not new_posts and not deleted_posts:
        print("No changes detected. All posts are up to date.")
        return

    # Deleted posts
    if deleted_posts:
        print(f"\nRemoving {len(deleted_posts)} deleted post(s)...")
        for post_id in deleted_posts:
            ids_to_remove = [
                cid for cid, c in chunk_store._chunks.items()
                if str(c.get("post_id")) == post_id
            ]
            if ids_to_remove:
                await vectorstore.remove(ids_to_remove)
                for cid in ids_to_remove:
                    del chunk_store._chunks[cid]
                chunk_store._persist()
            print(f"  Removed post {post_id} ({len(ids_to_remove)} chunks)")

    # New posts
    if new_posts:
        print(f"\nIndexing {len(new_posts)} new post(s)...")
        base = Path(doc_dir)

        for post_id in sorted(new_posts):
            post_dir = base / post_id
            data_json = post_dir / "data.json"
            if not data_json.exists():
                continue

            with open(data_json, encoding="utf-8") as f:
                metadata = json.load(f)

            title = metadata.get("title", "")
            print(f"  Post {post_id}: {title}")

            documents = []
            category = metadata.get("category", "unknown")
            main_file = metadata.get("main_file", "")
            attachments = metadata.get("attachments", [])
            all_supported = loader.SUPPORTED_EXTENSIONS | loader.IMAGE_EXTENSIONS
            year = metadata.get("year")
            url = metadata.get("url")

            main_path = post_dir / main_file
            if main_path.exists() and main_path.suffix.lower() in all_supported:
                try:
                    doc = loader.load_file(
                        str(main_path), category,
                        post_id=metadata.get("id"), post_title=title,
                        attachments=attachments, year=year, url=url,
                    )
                    documents.append(doc)
                except Exception as e:
                    print(f"    [SKIP] {main_path.name}: {e}")

            for att_name in attachments:
                att_path = post_dir / att_name
                if att_path.exists() and att_path.suffix.lower() in all_supported:
                    try:
                        doc = loader.load_file(
                            str(att_path), category,
                            post_id=metadata.get("id"), post_title=title,
                            year=year, url=url,
                        )
                        documents.append(doc)
                    except Exception as e:
                        print(f"    [SKIP] {att_path.name}: {e}")

            if not documents:
                continue

            chunks = chunker.chunk_documents(documents)
            print(f"    {len(documents)} file(s), {len(chunks)} chunks")

            if not chunks:
                continue

            chunk_texts = [c.text for c in chunks]
            chunk_embeddings = await embeddings.embed_batch(chunk_texts)
            chunk_ids = await vectorstore.upsert(chunks, chunk_embeddings)
            chunk_store.save_chunks(chunk_ids, chunks)

    # Rebuild BM25 if any changes
    if new_posts or deleted_posts:
        print("\nRebuilding BM25 index...")
        _rebuild_bm25(chunk_store, bm25)
        print("  BM25 index saved")

    print("\nIncremental indexing complete!")
    print(f"  New: {len(new_posts)} post(s)")
    print(f"  Deleted: {len(deleted_posts)} post(s)")
    print(f"  Total indexed: {len(chunk_store._chunks)} chunks")


async def add_post(post_id, loader, chunker, embeddings, vectorstore, chunk_store, bm25, doc_dir):
    """Add or re-index a specific post by ID."""
    chunk_store.load()
    base = Path(doc_dir)
    post_dir = base / str(post_id)

    if not post_dir.exists():
        print(f"Post folder not found: {post_dir}")
        return

    data_json = post_dir / "data.json"
    if not data_json.exists():
        print(f"data.json not found in {post_dir}")
        return

    with open(data_json, encoding="utf-8") as f:
        metadata = json.load(f)
    actual_id = str(metadata.get("id", post_id))

    existing_ids = [
        cid for cid, c in chunk_store._chunks.items()
        if str(c.get("post_id")) in (str(post_id), actual_id)
    ]
    if existing_ids:
        print(f"Removing existing {len(existing_ids)} chunks for post {post_id}...")
        await vectorstore.remove(existing_ids)
        for cid in existing_ids:
            del chunk_store._chunks[cid]
        chunk_store._persist()

    title = metadata.get("title", "")
    category = metadata.get("category", "unknown")
    main_file = metadata.get("main_file", "")
    attachments = metadata.get("attachments", [])
    all_supported = loader.SUPPORTED_EXTENSIONS | loader.IMAGE_EXTENSIONS

    print(f"Indexing post {post_id}: {title}")

    loader._load_image_cache(post_dir)
    year = metadata.get("year")
    url = metadata.get("url")

    documents = []
    main_path = post_dir / main_file
    if main_path.exists() and main_path.suffix.lower() in all_supported:
        doc = loader.load_file(
            str(main_path), category,
            post_id=metadata.get("id"), post_title=title,
            attachments=attachments, year=year, url=url,
        )
        documents.append(doc)

    for att_name in attachments:
        att_path = post_dir / att_name
        if att_path.exists() and att_path.suffix.lower() in all_supported:
            doc = loader.load_file(
                str(att_path), category,
                post_id=metadata.get("id"), post_title=title,
                year=year, url=url,
            )
            documents.append(doc)

    loader._save_image_cache()

    if not documents:
        print("  No documents found")
        return

    chunks = chunker.chunk_documents(documents)
    print(f"  {len(documents)} file(s), {len(chunks)} chunks")

    if not chunks:
        return

    chunk_texts = [c.text for c in chunks]
    chunk_embeddings = await embeddings.embed_batch(chunk_texts)
    chunk_ids = await vectorstore.upsert(chunks, chunk_embeddings)
    chunk_store.save_chunks(chunk_ids, chunks)

    _rebuild_bm25(chunk_store, bm25)
    print(f"  Done! Post {post_id} indexed ({len(chunks)} chunks)")


async def delete_post(post_id, vectorstore, chunk_store, bm25):
    """Delete a specific post from the index."""
    chunk_store.load()

    existing_ids = [
        cid for cid, c in chunk_store._chunks.items()
        if str(c.get("post_id")) == str(post_id)
    ]

    if not existing_ids:
        print(f"Post {post_id} not found in index.")
        return

    print(f"Deleting post {post_id} ({len(existing_ids)} chunks)...")
    await vectorstore.remove(existing_ids)
    for cid in existing_ids:
        del chunk_store._chunks[cid]
    chunk_store._persist()

    _rebuild_bm25(chunk_store, bm25)
    print(f"  Done! Post {post_id} deleted.")


async def _main_async(args):
    settings = get_settings()
    doc_dir = "./data/documents"

    loader = DocumentLoader()
    chunker = DocumentChunker(settings)
    vectorstore = VectorStore(settings)
    chunk_store = ChunkStore()
    bm25 = BM25Index()

    # add/delete 전용 가속: 임베딩 Redis 캐시 + BM25 토큰 캐시 로드.
    # full은 영향 X
    redis_client: Redis | None = None
    if (args.add or args.delete) and settings.redis_url:
        redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    if args.add or args.delete:
        bm25.load()

    embeddings = EmbeddingService(settings, redis_client=redis_client)

    try:
        if args.add:
            print(f"=== Add Post {args.add} ===")
            await add_post(args.add, loader, chunker, embeddings, vectorstore, chunk_store, bm25, doc_dir)
        elif args.delete:
            print(f"=== Delete Post {args.delete} ===")
            await delete_post(args.delete, vectorstore, chunk_store, bm25)
        elif args.full:
            print("=== Full Re-index ===")
            await full_index(loader, chunker, embeddings, vectorstore, chunk_store, bm25, doc_dir)
        else:
            print("=== Incremental Index ===")
            await incremental_index(loader, chunker, embeddings, vectorstore, chunk_store, bm25, doc_dir)
    finally:
        await vectorstore.close()
        if redis_client is not None:
            await redis_client.aclose()


def main():
    parser = argparse.ArgumentParser(description="Index documents into Firestore + BM25")
    parser.add_argument("--full", action="store_true", help="Full re-index")
    parser.add_argument("--add", type=str, help="Add/re-index a specific post by ID")
    parser.add_argument("--delete", type=str, help="Delete a specific post by ID")
    args = parser.parse_args()

    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
