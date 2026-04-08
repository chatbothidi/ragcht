import uuid

from google.cloud import aiplatform
import vertexai

from src.config import Settings
from src.models import DocumentChunk


class VectorStore:
    EMBEDDING_DIM = 768  # text-multilingual-embedding-002

    def __init__(self, settings: Settings):
        vertexai.init(
            project=settings.gcp_project_id,
            location=settings.gcp_location,
        )
        self.project = settings.gcp_project_id
        self.location = settings.gcp_location
        self.collection_name = settings.vertex_collection_name

        aiplatform.init(
            project=settings.gcp_project_id,
            location=settings.gcp_location,
        )

        # Auto-load existing index and endpoint if IDs are configured
        if settings.vertex_index_id and settings.vertex_endpoint_id:
            self.load_existing(settings.vertex_index_id, settings.vertex_endpoint_id)

    def create_collection(self) -> None:
        """Create Vector Search index and endpoint if they don't exist."""
        # Create index
        self.index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=self.collection_name,
            dimensions=self.EMBEDDING_DIM,
            approximate_neighbors_count=50,
            distance_measure_type="COSINE_DISTANCE",
            description="Medical and event document embeddings",
            index_update_method="STREAM_UPDATE",
        )

        # Create endpoint
        self.endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=f"{self.collection_name}-endpoint",
            public_endpoint_enabled=True,
        )

        # Deploy index to endpoint
        self.endpoint.deploy_index(
            index=self.index,
            deployed_index_id=self.collection_name.replace("-", "_") + "_v2",
            display_name=self.collection_name,
        )

    def load_existing(self, index_id: str, endpoint_id: str) -> None:
        """Load existing index and endpoint by resource IDs."""
        self.index = aiplatform.MatchingEngineIndex(index_name=index_id)
        self.endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=endpoint_id
        )

    def upsert(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> list[str]:
        """Upsert vectors with metadata to the index. Returns datapoint IDs."""
        datapoints = []
        ids = []
        for chunk, embedding in zip(chunks, embeddings):
            dp_id = str(uuid.uuid4())
            ids.append(dp_id)
            datapoints.append(
                {
                    "datapoint_id": dp_id,
                    "feature_vector": embedding,
                    "restricts": [
                        {
                            "namespace": "source_type",
                            "allow_list": [chunk.source_type],
                        },
                    ],
                    "crowding_tag": {"crowding_attribute": chunk.source_file},
                }
            )

        # Batch upsert
        batch_size = 100
        for i in range(0, len(datapoints), batch_size):
            batch = datapoints[i : i + batch_size]
            self.index.upsert_datapoints(datapoints=batch)

        return ids

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        source_type_filter: str | None = None,
    ) -> list[dict]:
        """Search for similar vectors."""
        from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
            Namespace,
        )

        filters = []
        if source_type_filter:
            filters.append(
                Namespace(name="source_type", allow_tokens=[source_type_filter])
            )

        response = self.endpoint.find_neighbors(
            deployed_index_id=self.collection_name.replace("-", "_") + "_v2",
            queries=[query_vector],
            num_neighbors=top_k,
            filter=filters if filters else None,
        )

        results = []
        if response and response[0]:
            for neighbor in response[0]:
                results.append(
                    {
                        "id": neighbor.id,
                        "score": 1.0 - neighbor.distance,  # cosine similarity
                    }
                )

        return results
