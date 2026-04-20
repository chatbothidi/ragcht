#!/usr/bin/env python3
"""Incremental document indexing into Vector Search + BM25.

Usage:
    python scripts/index_documents.py          # Incremental (new/deleted only)
    python scripts/index_documents.py --full   # Full re-index
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bm25_index import BM25Index
from src.chunk_store import ChunkStore
from src.chunker import DocumentChunker
from src.config import get_settings
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingService
from src.vectorstore import VectorStore


def full_index(loader, chunker, embeddings, vectorstore, chunk_store, bm25, doc_dir, settings=None):
    """Full re-index: clear everything and index all documents."""
    # Load all documents
    print(f"Loading documents from {doc_dir}...")
    documents = loader.load_directory(doc_dir)
    print(f"  Loaded {len(documents)} documents")

    if not documents:
        print("No documents found.")
        return

    medical_count = sum(1 for d in documents if d.source_type == "medical")
    event_count = sum(1 for d in documents if d.source_type == "event")
    print(f"  Medical: {medical_count}, Event: {event_count}")

    # Chunk
    print("\nChunking documents...")
    chunks = chunker.chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    # Embed
    print("\nGenerating embeddings (this may take a while)...")
    chunk_texts = [c.text for c in chunks]
    chunk_embeddings = embeddings.embed_batch(chunk_texts)
    print(f"  Generated {len(chunk_embeddings)} embeddings")

    # Clean ALL old data from Vector Search
    print("\nCleaning all old data from Vector Search...")
    from google.cloud import aiplatform as _aip

    endpoint = _aip.MatchingEngineIndexEndpoint(
        index_endpoint_name=settings.vertex_endpoint_id
    )
    deployed_id = settings.vertex_collection_name.replace("-", "_") + "_v2"

    # Use a generic query to find datapoints, then remove any not in new set
    dummy_vec = embeddings.embed("data", task_type="RETRIEVAL_QUERY")
    total_removed = 0
    while True:
        results = endpoint.find_neighbors(
            deployed_index_id=deployed_id,
            queries=[dummy_vec],
            num_neighbors=100,
        )
        if not results or not results[0]:
            break
        ids_to_remove = [n.id for n in results[0]]
        if not ids_to_remove:
            break
        try:
            vectorstore.index.remove_datapoints(datapoint_ids=ids_to_remove)
        except Exception:
            break
        total_removed += len(ids_to_remove)

    print(f"  Removed {total_removed} old datapoints")

    # Upload to Vector Search
    print("\nIndexing in Vertex AI Vector Search...")
    chunk_ids = vectorstore.upsert(chunks, chunk_embeddings)
    print("  Vector Search indexing complete")

    # Save chunk store
    print("\nSaving chunk store...")
    chunk_store._chunks = {}  # Reset
    chunk_store.save_chunks(chunk_ids, chunks)
    print("  Chunk store saved")

    # Build BM25
    print("\nBuilding BM25 index...")
    bm25.build(chunks)
    bm25.save()
    print("  BM25 index saved")

    print(f"\nFull indexing complete! Total chunks: {len(chunks)}")


def _scan_post_ids(doc_dir: str) -> set[str]:
    """Scan document directory for post IDs (folders with data.json)."""
    base = Path(doc_dir)
    post_ids = set()
    for d in base.iterdir():
        if d.is_dir() and (d / "data.json").exists():
            post_ids.add(d.name)
    return post_ids


def _get_indexed_post_ids(chunk_store: ChunkStore) -> set[str]:
    """Get post IDs from chunk store."""
    post_ids = set()
    for c in chunk_store._chunks.values():
        pid = c.get("post_id")
        if pid is not None:
            post_ids.add(str(pid))
    return post_ids


def incremental_index(loader, chunker, embeddings, vectorstore, chunk_store, bm25, doc_dir):
    """Incremental index: compare post IDs on disk vs chunk store."""
    import json

    chunk_store.load()

    # Compare post IDs
    disk_post_ids = _scan_post_ids(doc_dir)
    indexed_post_ids = _get_indexed_post_ids(chunk_store)

    new_posts = disk_post_ids - indexed_post_ids
    deleted_posts = indexed_post_ids - disk_post_ids

    if not new_posts and not deleted_posts:
        print("No changes detected. All posts are up to date.")
        return

    # Handle deleted posts
    if deleted_posts:
        print(f"\nRemoving {len(deleted_posts)} deleted post(s)...")
        for post_id in deleted_posts:
            # Find all chunk IDs for this post
            ids_to_remove = [
                cid for cid, c in chunk_store._chunks.items()
                if str(c.get("post_id")) == post_id
            ]
            if ids_to_remove:
                # Remove from Vector Search
                for i in range(0, len(ids_to_remove), 100):
                    batch = ids_to_remove[i : i + 100]
                    try:
                        vectorstore.index.remove_datapoints(datapoint_ids=batch)
                    except Exception:
                        pass
                # Remove from chunk store
                for cid in ids_to_remove:
                    del chunk_store._chunks[cid]
                chunk_store._persist()
            print(f"  Removed post {post_id} ({len(ids_to_remove)} chunks)")

    # Handle new posts
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

            # Load all documents from this post
            documents = []
            category = metadata.get("category", "unknown")
            main_file = metadata.get("main_file", "")
            attachments = metadata.get("attachments", [])
            all_supported = loader.SUPPORTED_EXTENSIONS | loader.IMAGE_EXTENSIONS

            # Main file
            main_path = post_dir / main_file
            if main_path.exists() and main_path.suffix.lower() in all_supported:
                try:
                    doc = loader.load_file(
                        str(main_path), category,
                        post_id=metadata.get("id"), post_title=title,
                        attachments=attachments,
                    )
                    documents.append(doc)
                except Exception as e:
                    print(f"    [SKIP] {main_path.name}: {e}")

            # Attachments
            for att_name in attachments:
                att_path = post_dir / att_name
                if att_path.exists() and att_path.suffix.lower() in all_supported:
                    try:
                        doc = loader.load_file(
                            str(att_path), category,
                            post_id=metadata.get("id"), post_title=title,
                        )
                        documents.append(doc)
                    except Exception as e:
                        print(f"    [SKIP] {att_path.name}: {e}")

            if not documents:
                continue

            # Chunk all documents from this post
            chunks = chunker.chunk_documents(documents)
            print(f"    {len(documents)} file(s), {len(chunks)} chunks")

            if not chunks:
                continue

            # Embed
            chunk_texts = [c.text for c in chunks]
            chunk_embeddings = embeddings.embed_batch(chunk_texts)

            # Upload to Vector Search
            chunk_ids = vectorstore.upsert(chunks, chunk_embeddings)

            # Save to chunk store
            chunk_store.save_chunks(chunk_ids, chunks)

    # Rebuild BM25
    if new_posts or deleted_posts:
        print("\nRebuilding BM25 index...")
        from src.models import DocumentChunk as DC
        all_chunks = [
            DC(
                text=c["text"],
                source_file=c["source_file"],
                source_type=c["source_type"],
                chunk_index=c["chunk_index"],
                page_number=c.get("page_number"),
                section_title=c.get("section_title"),
                metadata={"post_id": c.get("post_id"), "post_title": c.get("post_title")},
            )
            for c in chunk_store._chunks.values()
        ]
        bm25.build(all_chunks)
        bm25.save()
        print("  BM25 index saved")

    print(f"\nIncremental indexing complete!")
    print(f"  New: {len(new_posts)} post(s)")
    print(f"  Deleted: {len(deleted_posts)} post(s)")
    print(f"  Total indexed: {len(chunk_store._chunks)} chunks")


def add_post(post_id, loader, chunker, embeddings, vectorstore, chunk_store, bm25, doc_dir):
    """Add or re-index a specific post by ID."""
    import json

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

    # Read data.json to get the actual post ID
    with open(data_json, encoding="utf-8") as f:
        metadata = json.load(f)
    actual_id = str(metadata.get("id", post_id))

    # Remove existing chunks matching folder name OR data.json id
    existing_ids = [
        cid for cid, c in chunk_store._chunks.items()
        if str(c.get("post_id")) in (str(post_id), actual_id)
    ]
    if existing_ids:
        print(f"Removing existing {len(existing_ids)} chunks for post {post_id}...")
        for i in range(0, len(existing_ids), 100):
            batch = existing_ids[i : i + 100]
            try:
                vectorstore.index.remove_datapoints(datapoint_ids=batch)
            except Exception:
                pass
        for cid in existing_ids:
            del chunk_store._chunks[cid]
        chunk_store._persist()

    # Load and index
    title = metadata.get("title", "")
    category = metadata.get("category", "unknown")
    main_file = metadata.get("main_file", "")
    attachments = metadata.get("attachments", [])
    all_supported = loader.SUPPORTED_EXTENSIONS | loader.IMAGE_EXTENSIONS

    print(f"Indexing post {post_id}: {title}")

    # Load image cache
    loader._load_image_cache(post_dir)

    documents = []
    main_path = post_dir / main_file
    if main_path.exists() and main_path.suffix.lower() in all_supported:
        doc = loader.load_file(
            str(main_path), category,
            post_id=metadata.get("id"), post_title=title,
            attachments=attachments,
        )
        documents.append(doc)

    for att_name in attachments:
        att_path = post_dir / att_name
        if att_path.exists() and att_path.suffix.lower() in all_supported:
            doc = loader.load_file(
                str(att_path), category,
                post_id=metadata.get("id"), post_title=title,
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
    chunk_embeddings = embeddings.embed_batch(chunk_texts)
    chunk_ids = vectorstore.upsert(chunks, chunk_embeddings)
    chunk_store.save_chunks(chunk_ids, chunks)

    # Rebuild BM25
    from src.models import DocumentChunk as DC
    all_chunks = [
        DC(
            text=c["text"], source_file=c["source_file"], source_type=c["source_type"],
            chunk_index=c["chunk_index"], page_number=c.get("page_number"),
            section_title=c.get("section_title"),
            metadata={"post_id": c.get("post_id"), "post_title": c.get("post_title")},
        )
        for c in chunk_store._chunks.values()
    ]
    bm25.build(all_chunks)
    bm25.save()

    print(f"  Done! Post {post_id} indexed ({len(chunks)} chunks)")


def delete_post(post_id, vectorstore, chunk_store, bm25):
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

    for i in range(0, len(existing_ids), 100):
        batch = existing_ids[i : i + 100]
        try:
            vectorstore.index.remove_datapoints(datapoint_ids=batch)
        except Exception:
            pass

    for cid in existing_ids:
        del chunk_store._chunks[cid]
    chunk_store._persist()

    # Rebuild BM25
    from src.models import DocumentChunk as DC
    all_chunks = [
        DC(
            text=c["text"], source_file=c["source_file"], source_type=c["source_type"],
            chunk_index=c["chunk_index"], page_number=c.get("page_number"),
            section_title=c.get("section_title"),
            metadata={"post_id": c.get("post_id"), "post_title": c.get("post_title")},
        )
        for c in chunk_store._chunks.values()
    ]
    bm25.build(all_chunks)
    bm25.save()

    print(f"  Done! Post {post_id} deleted.")


def main():
    parser = argparse.ArgumentParser(description="Index documents")
    parser.add_argument("--full", action="store_true", help="Full re-index")
    parser.add_argument("--add", type=str, help="Add/re-index a specific post by ID")
    parser.add_argument("--delete", type=str, help="Delete a specific post by ID")
    args = parser.parse_args()

    settings = get_settings()
    doc_dir = "./data/documents"

    loader = DocumentLoader()
    chunker = DocumentChunker(settings)
    embeddings = EmbeddingService(settings)
    vectorstore = VectorStore(settings)
    chunk_store = ChunkStore()
    bm25 = BM25Index()

    if args.add:
        print(f"=== Add Post {args.add} ===")
        add_post(args.add, loader, chunker, embeddings, vectorstore, chunk_store, bm25, doc_dir)
    elif args.delete:
        print(f"=== Delete Post {args.delete} ===")
        delete_post(args.delete, vectorstore, chunk_store, bm25)
    elif args.full:
        print("=== Full Re-index ===")
        full_index(loader, chunker, embeddings, vectorstore, chunk_store, bm25, doc_dir, settings)
    else:
        print("=== Incremental Index ===")
        incremental_index(loader, chunker, embeddings, vectorstore, chunk_store, bm25, doc_dir)


if __name__ == "__main__":
    main()
