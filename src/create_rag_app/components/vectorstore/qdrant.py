"""
Qdrant vector store component.
"""
from textwrap import dedent
from create_rag_app.components.base import VectorStoreComponent

class QdrantComponent(VectorStoreComponent):
    """Implementation for the Qdrant vector store."""

    def get_docker_service(self) -> str:
        if self.deployment == "local":
            return dedent(f"""
            {self.service_name}:
                image: qdrant/qdrant:latest
                container_name: {self.service_name}
                restart: always
                ports:
                  - "6333:6333"
                  - "6334:6334"
                expose:
                  - 6333
                  - 6334
                volumes:
                  - ./qdrant_data:/qdrant/storage
                networks:
                  - app-network
            """).strip()
        return ""

    def get_env_vars(self) -> list[str]:
        if self.deployment == "cloud":
            return [
                'QDRANT_URL="your-qdrant-cloud-url"',
                'QDRANT_API_KEY="your-qdrant-api-key"'
            ]
        return ['QDRANT_URL="http://localhost:6333"']

    def get_requirements(self) -> list[str]:
        return ["qdrant-client", "langchain-qdrant"]

    def get_code_logic(self) -> str:
        if self.deployment == "cloud":
            return dedent("""
                self.client = QdrantClient(
                    url=config.qdrant_url,
                    api_key=Config.QDRANT_API_KEY
                )
            """).strip()
        return dedent("""
            self.client = QdrantClient(
                url=config.qdrant_url
            )
        """).strip()

    def get_collection_init_logic(self) -> str:
        return dedent(f"""
            try:
                # Try to create the collection regardless of whether it exists
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size={self.get_vector_dimension()},
                        distance=models.Distance.COSINE
                    ),
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            always_ram=True,
                        ),
                    ),
                )
                print(f"Collection '{{self.collection_name}}' created successfully!")
                
                # Update collection with HNSW parameters
                self.client.update_collection(
                    collection_name=self.collection_name,
                    hnsw_config=models.HnswConfigDiff(
                        m=64,
                        ef_construct=5000,
                        max_indexing_threads=8,
                        on_disk=True,
                    )
                )
            except Exception as e:
                # If collection already exists, this will handle that case
                if "already exists" in str(e):
                    print(f"Collection '{{self.collection_name}}' already exists.")
                else:
                    # Re-raise any other exceptions
                    raise
        """).strip()

    def get_config_class(self) -> str:
        return dedent("""
            class VectorStoreConfig(BaseModel):
                qdrant_url: str = Field(default=Config.QDRANT_URL, description="URL for Qdrant server")
                collection_name: str = Field(default=Config.COLLECTION_NAME, description="Name of the collection in Qdrant")
        """).strip()

    def get_vector_dimension(self) -> int:
        # This should be coordinated with the embedding model's dimension
        return 384  # Default for MiniLM
