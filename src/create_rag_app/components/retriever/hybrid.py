"""
Hybrid vector retrieval component.
"""
from textwrap import dedent
from ..base import RetrievalComponent

class HybridRetrievalComponent(RetrievalComponent):
    """Implementation for hybrid retrieval combining dense and sparse search."""

    def get_retrieval_imports(self) -> list[str]:
        return [
            "from langchain_qdrant import FastEmbedSparse, RetrievalMode"
        ]

    def get_retrieval_requirements(self) -> list[str]:
        return ["fastembed"]  # Required for sparse embeddings in hybrid mode

    def get_vectorstore_config_updates(self) -> str:
        """Return configuration updates needed for hybrid retrieval."""
        return dedent("""
            # Hybrid retrieval configuration
            sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")
        """).strip()

    def get_retrieval_init_logic(self, vectorstore_component) -> str:
        """Return initialization logic for hybrid retrieval."""
        vectorstore_id = vectorstore_component.config.get("id", "unknown")
        
        if vectorstore_id == "qdrant":
            return dedent("""
                # Hybrid retrieval initialization for Qdrant
                from langchain_qdrant import FastEmbedSparse, RetrievalMode
                
                # Initialize sparse embeddings for hybrid mode
                self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
                
                # Configure the vector store for hybrid retrieval
                self.vector_store.sparse_embedding = self.sparse_embeddings
                self.vector_store.retrieval_mode = RetrievalMode.HYBRID
                self.vector_store.vector_name = "dense"
                self.vector_store.sparse_vector_name = "sparse"
            """).strip()
        elif vectorstore_id == "pinecone":
            # For Pinecone, we can simulate hybrid using MMR (Maximum Marginal Relevance)
            return dedent("""
                # Hybrid retrieval simulation for Pinecone using MMR
                # MMR provides diversity which mimics some hybrid benefits
                pass
            """).strip()
        
        # For other vectorstores, fallback to dense
        return dedent("""
            # Hybrid retrieval not directly supported by this vector store
            # Falling back to dense retrieval
            print("Warning: Hybrid retrieval not supported, using dense retrieval")
        """).strip()

    def get_retrieval_method_logic(self, vectorstore_component) -> str:
        """Return the retrieval method implementation for hybrid search."""
        vectorstore_id = vectorstore_component.config.get("id", "unknown")
        
        if vectorstore_id == "qdrant":
            return dedent("""
                def retrieve(self, query: str, k: int = 5) -> list:
                    \"\"\"Retrieve documents using hybrid search (dense + sparse).\"\"\"
                    try:
                        # Use similarity search with hybrid retrieval mode
                        documents = self.vector_store.similarity_search(query, k=k)
                        return documents
                    except Exception as e:
                        print(f"Error during hybrid retrieval: {e}")
                        return []
                
                def retrieve_with_score(self, query: str, k: int = 5) -> list:
                    \"\"\"Retrieve documents with similarity scores using hybrid search.\"\"\"
                    try:
                        # Use similarity search with scores
                        documents_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
                        return documents_with_scores
                    except Exception as e:
                        print(f"Error during hybrid retrieval with scores: {e}")
                        return []
            """).strip()
        elif vectorstore_id == "pinecone":
            return dedent("""
                def retrieve(self, query: str, k: int = 5) -> list:
                    \"\"\"Retrieve documents using MMR for diversity (hybrid-like behavior).\"\"\"
                    try:
                        # Use MMR search for diversity, which provides hybrid-like benefits
                        documents = self.vector_store.max_marginal_relevance_search(query, k=k)
                        return documents
                    except Exception as e:
                        print(f"Error during hybrid retrieval (MMR): {e}")
                        # Fallback to regular similarity search
                        try:
                            documents = self.vector_store.similarity_search(query, k=k)
                            return documents
                        except Exception as fallback_e:
                            print(f"Error during fallback retrieval: {fallback_e}")
                            return []
                
                def retrieve_with_score(self, query: str, k: int = 5) -> list:
                    \"\"\"Retrieve documents with similarity scores.\"\"\"
                    try:
                        # Use similarity search with scores for Pinecone
                        documents_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
                        return documents_with_scores
                    except Exception as e:
                        print(f"Error during hybrid retrieval with scores: {e}")
                        return []
            """).strip()
        
        # For other vectorstores, fallback to dense
        return dedent("""
            def retrieve(self, query: str, k: int = 5) -> list:
                \"\"\"Retrieve documents using fallback dense search.\"\"\"
                try:
                    # Fallback to dense retrieval for unsupported vectorstores
                    documents = self.vector_store.similarity_search(query, k=k)
                    return documents
                except Exception as e:
                    print(f"Error during hybrid retrieval fallback: {e}")
                    return []
            
            def retrieve_with_score(self, query: str, k: int = 5) -> list:
                \"\"\"Retrieve documents with similarity scores using fallback dense search.\"\"\"
                try:
                    # Fallback to dense retrieval
                    documents_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
                    return documents_with_scores
                except Exception as e:
                    print(f"Error during hybrid retrieval fallback with scores: {e}")
                    return []
        """).strip()

    def supports_vectorstore(self, vectorstore_id: str) -> bool:
        """Hybrid retrieval is supported by Qdrant and partially by Pinecone."""
        return vectorstore_id in ["qdrant", "pinecone"]

    def get_search_method_name(self, vectorstore_id: str) -> str:
        """Get the appropriate search method name for hybrid retrieval."""
        if vectorstore_id == "pinecone":
            return "max_marginal_relevance_search"
        return "similarity_search" 