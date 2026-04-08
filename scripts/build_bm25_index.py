#!/usr/bin/env python3
"""Rebuild BM25 index from documents on disk."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bm25_index import BM25Index
from src.chunker import DocumentChunker
from src.config import get_settings
from src.document_loader import DocumentLoader


def main():
    settings = get_settings()

    loader = DocumentLoader()
    chunker = DocumentChunker(settings)
    bm25 = BM25Index()

    print("Loading documents...")
    documents = loader.load_directory("./data/documents")
    print(f"  Loaded {len(documents)} documents")

    print("Chunking...")
    chunks = chunker.chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    print("Building BM25 index...")
    bm25.build(chunks)
    bm25.save()
    print(f"  Saved to ./data/bm25_index/")

    print("Done!")


if __name__ == "__main__":
    main()
