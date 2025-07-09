"""
Dense vector search retrieval component.
"""
from textwrap import dedent
from typing import Dict, Any, List
from ..base import RetrievalComponent

class DenseRetrievalComponent(RetrievalComponent):
    """Implementation for dense vector search retrieval."""

    def get_env_vars(self) -> list[str]:
        """Dense retrieval doesn't need additional environment variables."""
        return []

    def get_requirements(self) -> list[str]:
        """Dense retrieval doesn't need additional requirements beyond langchain-qdrant."""
        return []

    def get_imports(self) -> list[str]:
        """Returns imports needed for dense retrieval."""
        base_imports = super().get_imports()
        base_imports.extend([
            "from langchain_qdrant import RetrievalMode"
        ])
        return base_imports

    def get_vectorstore_requirements(self) -> dict[str, any]:
        """Returns requirements for dense vector search."""
        return {
            "retrieval_mode": "RetrievalMode.DENSE",
            "needs_sparse_embedding": False,
            "additional_imports": ["from langchain_qdrant import RetrievalMode"],
            "additional_requirements": []
        }

    def get_retrieval_logic(self) -> str:
        """Returns the retrieval logic for dense vector search."""
        return dedent("""
            def retrieve(self, query: str, k: int = 4) -> List[Document]:
                \"\"\"Retrieve documents using dense vector similarity search.\"\"\"
                return self.vector_store.similarity_search(query, k=k)
            
            def retrieve_with_score(self, query: str, k: int = 4) -> List[tuple[Document, float]]:
                \"\"\"Retrieve documents with similarity scores using dense vectors.\"\"\"
                return self.vector_store.similarity_search_with_score(query, k=k)
        """).strip() 