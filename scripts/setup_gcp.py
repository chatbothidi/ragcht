#!/usr/bin/env python3
"""Create Vertex AI Vector Search collection and deploy endpoint."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.vectorstore import VectorStore


def main():
    settings = get_settings()

    print(f"Project: {settings.gcp_project_id}")
    print(f"Location: {settings.gcp_location}")
    print(f"Collection: {settings.vertex_collection_name}")
    print()

    store = VectorStore(settings)

    print("Creating Vector Search index and endpoint...")
    print("(This may take several minutes)")
    store.create_collection()

    print()
    print("Done! Save the following IDs for your .env:")
    print(f"  Index: {store.index.resource_name}")
    print(f"  Endpoint: {store.endpoint.resource_name}")


if __name__ == "__main__":
    main()
