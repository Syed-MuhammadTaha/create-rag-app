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
        if self.config.get("deployment") == "cloud":
            return dedent("""
            class VectorStoreConfig(BaseModel):
                qdrant_url: str = Field(
                    default=Config.QDRANT_URL, 
                    description="URL for Qdrant cloud server."
                )
                qdrant_api_key: str = Field(
                    default=Config.QDRANT_API_KEY,
                    description="API key for Qdrant cloud."
                )
                collection_name: str = Field(
                    default=Config.QDRANT_COLLECTION_NAME, 
                    description="Name of the collection in Qdrant."
                )
            """).strip()
        else:
            # Local deployment
            return dedent("""
            class VectorStoreConfig(BaseModel):
                qdrant_url: str = Field(
                    default=Config.QDRANT_URL, 
                    description="URL for local Qdrant server."
                )
                collection_name: str = Field(
                    default=Config.QDRANT_COLLECTION_NAME, 
                    description="Name of the collection in Qdrant."
                )
            """).strip()

    def get_init_logic(self) -> str:
        if self.config.get("deployment") == "cloud":
            return dedent("""
                self.embeddings = Embedder()
                
                # Initialize Qdrant client for cloud deployment
                self.client = QdrantClient(
                    url=config.qdrant_url,
                    api_key=config.qdrant_api_key
                )
                
                self.collection_name = config.collection_name
                self.initialize_collection()
                
                self.vector_store = QdrantVectorStore(
                    client=self.client,
                    collection_name=self.collection_name,
                    embedding=self.embeddings
                )
            """).strip()
        else:
            # Local deployment
            return dedent("""
                self.embeddings = Embedder()
                
                # Initialize Qdrant client for local deployment
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
                # Check if we need sparse vectors (for sparse or hybrid retrieval)
                retrieval_mode = getattr(self, 'retrieval_mode', 'DENSE')
                
                if 'SPARSE' in str(retrieval_mode) or 'HYBRID' in str(retrieval_mode):
                    # Create collection with both dense and sparse vectors
                    from qdrant_client.http.models import SparseVectorParams
                    from qdrant_client import models
                    
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config={
                            "dense": VectorParams(
                                size=self.embeddings.get_vector_dimension(),
                                distance=Distance.COSINE
                            )
                        },
                        sparse_vectors_config={
                            "sparse": SparseVectorParams(
                                index=models.SparseIndexParams(on_disk=False)
                            )
                        }
                    )
                    print(f"Collection '{self.collection_name}' created with dense and sparse vectors.")
                else:
                    # Create collection with only dense vectors
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=self.embeddings.get_vector_dimension(),
                            distance=Distance.COSINE
                        )
                    )
                    print(f"Collection '{self.collection_name}' created with dense vectors.")
            else:
                print(f"Collection '{self.collection_name}' already exists.")
        """).strip()
