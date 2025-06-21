"""
Qdrant vector store component.
"""
from textwrap import dedent
from ..base import VectorStoreComponent, ProvidesDockerService

class QdrantComponent(VectorStoreComponent, ProvidesDockerService):
    """Implementation for the Qdrant vector store."""

    @property
    def service_name(self) -> str:
        """The name of the Docker service for this component."""
        return "qdrant-vectorstore"

    def get_docker_service(self) -> str:
        if self.config.get("deployment") == "local":
            return dedent(f"""
            {self.service_name}:
                image: qdrant/qdrant:v1.7.4
                container_name: {self.service_name}
                ports:
                  - "6333:6333"
                  - "6334:6334"
                volumes:
                  - qdrant_data:/qdrant/storage
                networks:
                  - app-network
            """).strip()
        return ""

    def get_env_vars(self) -> list[str]:
        if self.config.get("deployment") == "cloud":
            return [
                'QDRANT_URL="your-qdrant-cloud-url"',
                'QDRANT_API_KEY="your-qdrant-api-key"',
                'QDRANT_COLLECTION_NAME="your-collection-name"'
            ]
        # For local deployment
        return [
            'QDRANT_URL="http://qdrant-vectorstore:6333"',
            'QDRANT_COLLECTION_NAME="rag-app-collection"'
        ]

    def get_requirements(self) -> list[str]:
        return ["qdrant-client==1.7.3"]

    def get_imports(self) -> list[str]:
        base_imports = super().get_imports()
        base_imports.extend([
            "from qdrant_client import QdrantClient, models"
        ])
        return base_imports

    def get_config_class(self) -> str:
        return dedent("""
        class VectorStoreConfig(BaseModel):
            qdrant_url: str = Field(
                default=Config.QDRANT_URL, 
                description="URL for Qdrant server."
            )
            collection_name: str = Field(
                default=Config.QDRANT_COLLECTION_NAME, 
                description="Name of the collection in Qdrant."
            )
        """).strip()

    def get_code_logic(self) -> str:
        return dedent("""
            # Initialize Qdrant client
            if Config.QDRANT_API_KEY:
                self.client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
            else:
                self.client = QdrantClient(url=Config.QDRANT_URL)

            # Create or verify the collection
            try:
                self.client.get_collection(collection_name=Config.QDRANT_COLLECTION_NAME)
                print(f"Collection '{Config.QDRANT_COLLECTION_NAME}' already exists.")
            except Exception:
                print(f"Collection '{Config.QDRANT_COLLECTION_NAME}' not found. Creating a new one...")
                self.client.create_collection(
                    collection_name=Config.QDRANT_COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=self.embeddings.get_vector_dimension(),
                        distance=models.Distance.COSINE
                    )
                )
                print("Collection created successfully.")
        """).strip()

    def get_vector_dimension(self) -> int:
        # This should be coordinated with the embedding model's dimension
        return 384  # Default for MiniLM
