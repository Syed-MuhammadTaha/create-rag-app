"""
Sparse vector retrieval component.
"""
from textwrap import dedent
from ..base import RetrievalComponent

class SparseRetrievalComponent(RetrievalComponent):
    """Implementation for sparse vector retrieval using keyword-based search."""

    def get_retrieval_imports(self) -> list[str]:
        return [
            "from langchain_qdrant import FastEmbedSparse, RetrievalMode"
        ]

    def get_retrieval_requirements(self) -> list[str]:
        return ["fastembed"]  # Required for sparse embeddings

    def get_vectorstore_config_updates(self) -> str:
        """Return configuration updates needed for sparse retrieval."""
        return dedent("""
            # Sparse retrieval configuration
            sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")
        """).strip()

    def get_retrieval_init_logic(self, vectorstore_component) -> str:
        """Return initialization logic for sparse retrieval."""
        vectorstore_id = vectorstore_component.config.get("id", "unknown")
        
        if vectorstore_id == "qdrant":
            return dedent("""
                # Sparse retrieval initialization for Qdrant
                from langchain_qdrant import FastEmbedSparse, RetrievalMode
                
                # Initialize sparse embeddings
                self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
                
                # Configure the vector store for sparse retrieval
                self.vector_store.sparse_embedding = self.sparse_embeddings
                self.vector_store.retrieval_mode = RetrievalMode.SPARSE
                self.vector_store.sparse_vector_name = "sparse"
            """).strip()
        
        # Sparse retrieval is primarily supported by Qdrant
        return dedent("""
            # Sparse retrieval is not directly supported by this vector store
            # Falling back to dense retrieval
            print("Warning: Sparse retrieval not supported, using dense retrieval")
        """).strip()

    def get_retrieval_method_logic(self, vectorstore_component) -> str:
        """Return the retrieval method implementation for sparse search."""
        vectorstore_id = vectorstore_component.config.get("id", "unknown")
        
        if vectorstore_id == "qdrant":
            return dedent("""
                def retrieve(self, query: str, k: int = 5) -> list:
                    \"\"\"Retrieve documents using sparse vector search.\"\"\"
                    try:
                        # Use similarity search with sparse retrieval mode
                        documents = self.vector_store.similarity_search(query, k=k)
                        return documents
                    except Exception as e:
                        print(f"Error during sparse retrieval: {e}")
                        return []
                
                def retrieve_with_score(self, query: str, k: int = 5) -> list:
                    \"\"\"Retrieve documents with similarity scores using sparse search.\"\"\"
                    try:
                        # Use similarity search with scores
                        documents_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
                        return documents_with_scores
                    except Exception as e:
                        print(f"Error during sparse retrieval with scores: {e}")
                        return []
            """).strip()
        
        # For non-Qdrant vectorstores, fallback to dense
        return dedent("""
            def retrieve(self, query: str, k: int = 5) -> list:
                \"\"\"Retrieve documents using fallback dense search.\"\"\"
                try:
                    # Fallback to dense retrieval for non-Qdrant vectorstores
                    documents = self.vector_store.similarity_search(query, k=k)
                    return documents
                except Exception as e:
                    print(f"Error during sparse retrieval fallback: {e}")
                    return []
            
            def retrieve_with_score(self, query: str, k: int = 5) -> list:
                \"\"\"Retrieve documents with similarity scores using fallback dense search.\"\"\"
                try:
                    # Fallback to dense retrieval
                    documents_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
                    return documents_with_scores
                except Exception as e:
                    print(f"Error during sparse retrieval fallback with scores: {e}")
                    return []
        """).strip()

    def supports_vectorstore(self, vectorstore_id: str) -> bool:
        """Sparse retrieval is primarily supported by Qdrant."""
        return vectorstore_id == "qdrant"

    def get_search_method_name(self, vectorstore_id: str) -> str:
        """Get the appropriate search method name for sparse retrieval."""
        return "similarity_search" 