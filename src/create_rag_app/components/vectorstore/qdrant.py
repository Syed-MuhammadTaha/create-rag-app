"""
Qdrant vector store component.
"""
from textwrap import dedent
from ..base import VectorStoreComponent, ProvidesDockerService

class QdrantComponent(VectorStoreComponent, ProvidesDockerService):
    """Implementation for the Qdrant vector store."""

    def __init__(self, config, retrieval_requirements=None):
        """Initialize with config and optional retrieval requirements."""
        super().__init__(config)
        self.retrieval_requirements = retrieval_requirements or {}

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
        requirements = ["qdrant-client", "langchain-qdrant"]
        
        # Add retrieval-specific requirements
        additional_requirements = self.retrieval_requirements.get("additional_requirements", [])
        requirements.extend(additional_requirements)
        
        return requirements

    def get_imports(self) -> list[str]:
        base_imports = super().get_imports()
        base_imports.extend([
            "from qdrant_client import QdrantClient",
            "from qdrant_client.http.models import Distance, VectorParams",
            "from langchain_qdrant import QdrantVectorStore",
            "from langchain_core.documents import Document"
        ])
        
        # Add retrieval-specific imports
        additional_imports = self.retrieval_requirements.get("additional_imports", [])
        base_imports.extend(additional_imports)
        
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
        # Build client initialization based on deployment
        if self.config.get("deployment") == "cloud":
            client_init = dedent("""
                # Initialize Qdrant client for cloud deployment
                self.client = QdrantClient(
                    url=config.qdrant_url,
                    api_key=config.qdrant_api_key
                )""")
        else:
            client_init = dedent("""
                # Initialize Qdrant client for local deployment
                self.client = QdrantClient(url=config.qdrant_url)""")
        
        # Build vector store configuration based on retrieval requirements
        retrieval_mode = self.retrieval_requirements.get("retrieval_mode", "RetrievalMode.DENSE")
        vector_store_config = {
            "client": "self.client",
            "collection_name": "self.collection_name",
            "embedding": "self.embeddings",
            "retrieval_mode": retrieval_mode
        }
        
        # Add sparse embedding if needed
        if self.retrieval_requirements.get("needs_sparse_embedding"):
            sparse_setup = self.retrieval_requirements.get("sparse_embedding_setup", "")
            vector_store_config["sparse_embedding"] = f"sparse_embeddings"
            sparse_init = f"sparse_embeddings = {sparse_setup}"
        else:
            sparse_init = ""
        
        # Add vector names if specified
        if "vector_name" in self.retrieval_requirements:
            vector_store_config["vector_name"] = f'"{self.retrieval_requirements["vector_name"]}"'
        
        if "sparse_vector_name" in self.retrieval_requirements:
            vector_store_config["sparse_vector_name"] = f'"{self.retrieval_requirements["sparse_vector_name"]}"'
        
        # Build the vector store initialization
        vector_store_params = ",\n    ".join([f"{k}={v}" for k, v in vector_store_config.items()])
        
        init_logic = f"""
self.embeddings = Embedder()
{client_init}

self.collection_name = config.collection_name
self.initialize_collection()
{sparse_init}

self.vector_store = QdrantVectorStore(
    {vector_store_params}
)
"""
        return dedent(init_logic).strip()

    def get_initialize_collection_logic(self) -> str:
        if self.retrieval_requirements.get("requires_sparse_vectors_config"):
            # Hybrid/sparse retrieval needs both dense and sparse vector config
            return dedent("""
                collections = [c.name for c in self.client.get_collections().collections]
                if self.collection_name not in collections:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config={"dense": VectorParams(size=self.embeddings.get_vector_dimension(), distance=Distance.COSINE)},
                        sparse_vectors_config={
                            "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
                        }
                    )
                    print(f"Collection '{self.collection_name}' created successfully with dense and sparse vectors.")
                else:
                    print(f"Collection '{self.collection_name}' already exists.")
            """).strip()
        else:
            # Dense-only retrieval
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
