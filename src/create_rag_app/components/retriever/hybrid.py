"""
Hybrid vector search retrieval component.
"""
from textwrap import dedent
from typing import Dict, Any, List
from ..base import RetrievalComponent

class HybridRetrievalComponent(RetrievalComponent):
    """Implementation for hybrid vector search retrieval combining dense and sparse vectors."""

    def get_env_vars(self) -> list[str]:
        """Hybrid retrieval doesn't need additional environment variables."""
        return []

    def get_requirements(self) -> list[str]:
        """Hybrid retrieval needs FastEmbed for sparse embeddings."""
        return ["fastembed"]

    def get_imports(self) -> list[str]:
        """Returns imports needed for hybrid retrieval."""
        base_imports = super().get_imports()
        base_imports.extend([
            "from langchain_qdrant import FastEmbedSparse, RetrievalMode",
            "from qdrant_client import models",
            "from qdrant_client.http.models import SparseVectorParams"
        ])
        return base_imports

    def get_vectorstore_requirements(self) -> dict[str, any]:
        """Returns requirements for hybrid vector search."""
        return {
            "retrieval_mode": "RetrievalMode.HYBRID",
            "needs_sparse_embedding": True,
            "sparse_embedding_setup": 'FastEmbedSparse(model_name="Qdrant/bm25")',
            "vector_name": "dense",
            "sparse_vector_name": "sparse",
            "additional_imports": [
                "from langchain_qdrant import FastEmbedSparse, RetrievalMode",
                "from qdrant_client import models",
                "from qdrant_client.http.models import SparseVectorParams"
            ],
            "additional_requirements": ["fastembed"],
            "requires_sparse_vectors_config": True
        }

    def get_retrieval_logic(self) -> str:
        """Returns the retrieval logic for hybrid vector search."""
        return dedent("""
            def retrieve(self, query: str, k: int = 4) -> List[Document]:
                \"\"\"Retrieve documents using hybrid (dense + sparse) vector search.\"\"\"
                return self.vector_store.similarity_search(query, k=k)
            
            def retrieve_with_score(self, query: str, k: int = 4) -> List[tuple[Document, float]]:
                \"\"\"Retrieve documents with similarity scores using hybrid vectors.\"\"\"
                return self.vector_store.similarity_search_with_score(query, k=k)
        """).strip() 