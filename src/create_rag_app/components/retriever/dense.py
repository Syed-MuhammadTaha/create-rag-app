"""
Dense vector retrieval component.
"""
from textwrap import dedent
from ..base import RetrievalComponent

class DenseRetrievalComponent(RetrievalComponent):
    """Implementation for dense vector retrieval."""

    def get_retrieval_imports(self) -> list[str]:
        return []  # No additional imports needed for dense retrieval

    def get_retrieval_requirements(self) -> list[str]:
        return []  # No additional requirements for dense retrieval

    def get_vectorstore_config_updates(self) -> str:
        """Dense retrieval doesn't need special vectorstore configuration."""
        return ""

    def get_retrieval_init_logic(self, vectorstore_component) -> str:
        """Return initialization logic for dense retrieval."""
        vectorstore_id = vectorstore_component.config.get("id", "unknown")
        
        if vectorstore_id == "qdrant":
            return dedent("""
                # Dense retrieval initialization for Qdrant
                from langchain_qdrant import RetrievalMode
                
                # Configure the vector store for dense retrieval
                self.vector_store.retrieval_mode = RetrievalMode.DENSE
            """).strip()
        
        # For Pinecone and other vectorstores, dense is the default
        return dedent("""
            # Dense retrieval is the default mode for most vector stores
            pass
        """).strip()

    def get_retrieval_method_logic(self, vectorstore_component) -> str:
        """Return the retrieval method implementation for dense search."""
        return dedent("""
            def retrieve(self, query: str, k: int = 5) -> list:
                \"\"\"Retrieve documents using dense vector similarity search.\"\"\"
                try:
                    # Use similarity search for dense retrieval
                    documents = self.vector_store.similarity_search(query, k=k)
                    return documents
                except Exception as e:
                    print(f"Error during dense retrieval: {e}")
                    return []
            
            def retrieve_with_score(self, query: str, k: int = 5) -> list:
                \"\"\"Retrieve documents with similarity scores.\"\"\"
                try:
                    # Use similarity search with scores
                    documents_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
                    return documents_with_scores
                except Exception as e:
                    print(f"Error during dense retrieval with scores: {e}")
                    return []
        """).strip()

    def supports_vectorstore(self, vectorstore_id: str) -> bool:
        """Dense retrieval supports all vectorstores."""
        return True

    def get_search_method_name(self, vectorstore_id: str) -> str:
        """Get the appropriate search method name for dense retrieval."""
        return "similarity_search" 