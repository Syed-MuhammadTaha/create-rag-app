"""
Pinecone vector store component.
"""
from textwrap import dedent
from create_rag_app.components.base import VectorStoreComponent

class PineconeComponent(VectorStoreComponent):
    """Implementation for the Pinecone vector store."""

    def get_docker_service(self) -> str:
        # Pinecone is cloud-only, no Docker service needed
        return ""

    def get_env_vars(self) -> list[str]:
        return [
            'PINECONE_API_KEY="your-pinecone-api-key"',
            'PINECONE_ENVIRONMENT="your-pinecone-environment"',
            'PINECONE_INDEX="your-pinecone-index-name"'
        ]

    def get_requirements(self) -> list[str]:
        return ["pinecone-client", "langchain-pinecone"]

    def get_code_logic(self) -> str:
        return dedent("""
            import pinecone

            # Initialize Pinecone
            pinecone.init(
                api_key=Config.PINECONE_API_KEY,
                environment=Config.PINECONE_ENVIRONMENT
            )
            self.index = pinecone.Index(Config.PINECONE_INDEX)
        """).strip()

    def get_collection_init_logic(self) -> str:
        return dedent(f"""
            try:
                # Check if index exists
                if Config.PINECONE_INDEX not in pinecone.list_indexes():
                    # Create index if it doesn't exist
                    pinecone.create_index(
                        name=Config.PINECONE_INDEX,
                        dimension={self.get_vector_dimension()},
                        metric="cosine"
                    )
                    print(f"Index '{{Config.PINECONE_INDEX}}' created successfully!")
                else:
                    print(f"Index '{{Config.PINECONE_INDEX}}' already exists.")
            except Exception as e:
                print(f"Error initializing Pinecone index: {{e}}")
                raise
        """).strip()

    def get_config_class(self) -> str:
        return dedent("""
            class VectorStoreConfig(BaseModel):
                index_name: str = Field(default=Config.PINECONE_INDEX, description="Name of the Pinecone index")
                namespace: str = Field(default="default", description="Namespace within the Pinecone index")
        """).strip()

    def get_vector_dimension(self) -> int:
        # This should be coordinated with the embedding model's dimension
        return 384  # Default for MiniLM
