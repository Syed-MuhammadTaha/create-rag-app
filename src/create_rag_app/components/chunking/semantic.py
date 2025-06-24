from textwrap import dedent
from ..base import ChunkingComponent

class SemanticChunkingComponent(ChunkingComponent):
    """Implementation for semantic chunking."""

    def get_env_vars(self) -> list[str]:
        return []

    def get_requirements(self) -> list[str]:
        return ["langchain-experimental", "sentence-transformers"]

    def get_imports(self) -> list[str]:
        base_imports = super().get_imports()
        base_imports.extend([
            "from langchain_experimental.text_splitter import SemanticChunker",
            "from src.utils.embedder import Embedder"
        ])
        return base_imports

    def get_config_class(self) -> str:
        return dedent("""
        class ChunkingConfig(BaseModel):
            breakpoint_threshold_type: str = Field(default="percentile", description="Threshold type for semantic chunking.")
        """).strip()

    def get_code_logic(self) -> str:
        return dedent("""
        @classmethod
        def split_text(cls, text: str, config: ChunkingConfig = ChunkingConfig()):
            \"\"\"Splits text into chunks using semantic boundaries.\"\"\"
            text_splitter = SemanticChunker(
                Embedder(), breakpoint_threshold_type=config.breakpoint_threshold_type
            )
            # SemanticChunker returns Document objects, so we extract the page_content.
            documents = text_splitter.create_documents([text])
            return [doc.page_content for doc in documents]
        """).strip() 