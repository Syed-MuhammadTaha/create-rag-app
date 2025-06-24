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
                image: qdrant/qdrant:v1.12.5
                container_name: {self.service_name}
                ports:
                  - 6333:6333
                  - 6334:6334
                expose:
                  - 6333
                  - 6334
                  - 6335
                volumes:
                  - ./qdrant_data:/qdrant/storage
                networks:
                  - app-network
            """).strip()
        return ""

    def get_env_vars(self) -> list[str]:
        if self.config.get("deployment") == "cloud":
            return [
                'QDRANT_URL="your-qdrant-cloud-url"',
                'QDRANT_API_KEY="your-qdrant-api-key"',
                'QDRANT_COLLECTION_NAME="rag-db"'
            ]
        # For local deployment
        return [
            'QDRANT_URL="http://qdrant-vectorstore:6333"',
            'QDRANT_COLLECTION_NAME="rag-db"'
        ]

    def get_requirements(self) -> list[str]:
        return ["qdrant-client", "langchain-qdrant"]

    def get_imports(self) -> list[str]:
        base_imports = super().get_imports()
        base_imports.extend([
            "from qdrant_client import QdrantClient",
            "from qdrant_client.http.models import Distance, VectorParams",
            "from langchain_qdrant import QdrantVectorStore"
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

    def get_init_logic(self) -> str:
        return dedent("""
            self.embeddings = Embedder()
            
            # Initialize Qdrant client
            if hasattr(config, 'qdrant_api_key') and config.qdrant_api_key:
                self.client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
            else:
                self.client = QdrantClient(url=config.qdrant_url)
            
            self.collection_name = config.collection_name
            self.initialize_collection()
            
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings
            )
        """).strip()

    def get_initialize_collection_logic(self) -> str:
        return dedent("""
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embeddings.get_vector_dimension(),
                        distance=Distance.COSINE
                    )
                )
                print(f"Collection '{self.collection_name}' created successfully.")
            else:
                print(f"Collection '{self.collection_name}' already exists.")
        """).strip()
