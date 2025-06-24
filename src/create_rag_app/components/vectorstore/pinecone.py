"""
Pinecone vector store component.
"""
from textwrap import dedent
from ..base import VectorStoreComponent

class PineconeComponent(VectorStoreComponent):
    """Implementation for the Pinecone vector store."""

    def get_env_vars(self) -> list[str]:
        return [
            'PINECONE_API_KEY="your-pinecone-api-key"',
            'PINECONE_INDEX_NAME="rag-db"'
        ]

    def get_requirements(self) -> list[str]:
        return ["langchain-pinecone", "pinecone-client"]

    def get_imports(self) -> list[str]:
        # Extends the base imports with pinecone-specific ones
        base_imports = super().get_imports()
        base_imports.extend([
            "from langchain_pinecone import PineconeVectorStore",
            "from pinecone import Pinecone, ServerlessSpec"
        ])
        return base_imports

    def get_config_class(self) -> str:
        return dedent("""
        class VectorStoreConfig(BaseModel):
            pinecone_api_key: str = Field(default=Config.PINECONE_API_KEY, description="Pinecone API key")
            index_name: str = Field(default=Config.PINECONE_INDEX_NAME, description="Name of the collection in Pinecone")
        """).strip()
    
    def get_init_logic(self) -> str:
        return dedent("""
            self.embeddings = Embedder()
            self.pc = Pinecone(api_key=config.pinecone_api_key)
            self.index_name = config.index_name
            
            # Create collection if it doesn't exist
            self.initialize_collection()

            index = self.pc.Index(self.index_name)
            
            # Initialize the vector store interface
            self.vector_store = PineconeVectorStore(
                index=index,
                embedding=self.embeddings
            )
        """).strip()

    def get_initialize_collection_logic(self) -> str:
        return dedent("""
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embeddings.get_vector_dimension(),
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                print(f"Collection '{self.index_name}' created successfully!")
            else:
                print(f"Collection '{self.index_name}' already exists.")
        """).strip()
