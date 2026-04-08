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


def full_index(loader, chunker, embeddings, vectorstore, chunk_store, bm25, doc_dir):
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


def incremental_index(loader, chunker, embeddings, vectorstore, chunk_store, bm25, doc_dir):
    """Incremental index: only process new/deleted documents."""
    # Load chunk store to get already indexed files
    chunk_store.load()
    indexed_files = chunk_store.get_indexed_files()

    # Scan current files on disk
    current_files: dict[str, tuple[str, str]] = {}  # filename -> (path, source_type)
    base_path = Path(doc_dir)
    for subdir in ["medical", "events"]:
        subdir_path = base_path / subdir
        if not subdir_path.exists():
            continue
        source_type = "medical" if subdir == "medical" else "event"
        for file_path in subdir_path.rglob("*"):
            if file_path.suffix.lower() in loader.SUPPORTED_EXTENSIONS:
                current_files[file_path.name] = (str(file_path), source_type)

    current_file_names = set(current_files.keys())

    # Determine new and deleted files
    new_files = current_file_names - indexed_files
    deleted_files = indexed_files - current_file_names

    if not new_files and not deleted_files:
        print("No changes detected. All documents are up to date.")
        return

    # Handle deleted files
    if deleted_files:
        print(f"\nRemoving {len(deleted_files)} deleted document(s)...")
        for filename in deleted_files:
            removed_ids = chunk_store.remove_by_source(filename)
            if removed_ids:
                for i in range(0, len(removed_ids), 100):
                    batch = removed_ids[i : i + 100]
                    try:
                        vectorstore.index.remove_datapoints(datapoint_ids=batch)
                    except Exception:
                        pass
            print(f"  Removed: {filename} ({len(removed_ids)} chunks)")

    # Handle new files
    if new_files:
        print(f"\nIndexing {len(new_files)} new document(s)...")
        for filename in new_files:
            file_path, source_type = current_files[filename]
            print(f"  Processing: {filename}")

            # Load
            doc = loader.load_file(file_path, source_type)

            # Chunk
            chunks = chunker.chunk_document(doc)
            print(f"    Created {len(chunks)} chunks")

            if not chunks:
                continue

            # Embed
            chunk_texts = [c.text for c in chunks]
            chunk_embeddings = embeddings.embed_batch(chunk_texts)

            # Upload to Vector Search
            chunk_ids = vectorstore.upsert(chunks, chunk_embeddings)

            # Save to chunk store
            chunk_store.save_chunks(chunk_ids, chunks)

            print(f"    Indexed {len(chunks)} chunks")

    # Rebuild BM25 from chunk store (BM25 doesn't support incremental well)
    if new_files or deleted_files:
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
            )
            for c in chunk_store._chunks.values()
        ]
        bm25.build(all_chunks)
        bm25.save()
        print("  BM25 index saved")

    print(f"\nIncremental indexing complete!")
    print(f"  New: {len(new_files)} document(s)")
    print(f"  Deleted: {len(deleted_files)} document(s)")
    print(f"  Total indexed: {len(chunk_store._chunks)} chunks")


def main():
    parser = argparse.ArgumentParser(description="Index documents")
    parser.add_argument("--full", action="store_true", help="Full re-index")
    args = parser.parse_args()

    settings = get_settings()
    doc_dir = "./data/documents"

    loader = DocumentLoader()
    chunker = DocumentChunker(settings)
    embeddings = EmbeddingService(settings)
    vectorstore = VectorStore(settings)
    chunk_store = ChunkStore()
    bm25 = BM25Index()

    if args.full:
        print("=== Full Re-index ===")
        full_index(loader, chunker, embeddings, vectorstore, chunk_store, bm25, doc_dir)
    else:
        print("=== Incremental Index ===")
        incremental_index(loader, chunker, embeddings, vectorstore, chunk_store, bm25, doc_dir)


if __name__ == "__main__":
    main()
