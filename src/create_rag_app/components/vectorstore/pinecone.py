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
            'PINECONE_INDEX_NAME="your-pinecone-index-name"'
        ]

    def get_requirements(self) -> list[str]:
        return ["pinecone-client==2.2.4"]

    def get_imports(self) -> list[str]:
        # Extends the base imports with pinecone-specific ones
        base_imports = super().get_imports()
        base_imports.extend([
            "pinecone"
        ])
        return base_imports

    def get_config_class(self) -> str:
        return dedent("""
        class VectorStoreConfig(BaseModel):
            pinecone_index_name: str = Field(
                default=Config.PINECONE_INDEX_NAME, 
                description="Name of the Pinecone index."
            )
        """).strip()

    def get_code_logic(self) -> str:
        return dedent("""
            self.pc = pinecone.Pinecone(api_key=Config.PINECONE_API_KEY)
            
            # Initialize the index
            print(f"Initializing Pinecone index: {Config.PINECONE_INDEX_NAME}...")
            if Config.PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
                print(f"Index not found. Creating a new index...")
                self.pc.create_index(
                    name=Config.PINECONE_INDEX_NAME,
                    dimension=self.embeddings.get_vector_dimension(),
                    metric="cosine"
                )
                print("Index created successfully.")
            else:
                print("Index already exists.")
            
            self.index = self.pc.Index(Config.PINECONE_INDEX_NAME)
        """).strip()
