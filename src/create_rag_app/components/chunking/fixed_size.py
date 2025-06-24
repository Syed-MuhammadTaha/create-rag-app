from textwrap import dedent
from ..base import ChunkingComponent

class FixedSizeChunkingComponent(ChunkingComponent):
    """Implementation for fixed-size chunking."""

    def get_env_vars(self) -> list[str]:
        return []

    def get_requirements(self) -> list[str]:
        return ["langchain-text-splitters"]

    def get_imports(self) -> list[str]:
        base_imports = super().get_imports()
        base_imports.extend([
            "from langchain_text_splitters import RecursiveCharacterTextSplitter"
        ])
        return base_imports

    def get_config_class(self) -> str:
        return dedent("""
        class ChunkingConfig(BaseModel):
            chunk_size: int = Field(default=1500, description="Size of each text chunk.")
            chunk_overlap: int = Field(default=150, description="Overlap between consecutive chunks.")
        """).strip()

    def get_code_logic(self) -> str:
        return dedent("""
        @classmethod
        def split_text(cls, text: str, config: ChunkingConfig = ChunkingConfig()):
            \"\"\"Splits text into chunks for better processing.\"\"\"
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
            return text_splitter.split_text(text)
        """).strip() 