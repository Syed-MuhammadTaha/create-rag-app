"""
Base classes and mixins for RAG application components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

# --- Core Base Class ---

class BaseComponent(ABC):
    """Minimal abstract base class for all components."""

    def __init__(self, config: Dict[str, Any]):
        """Initializes the component with its specific configuration."""
        self.config = config

    @property
    def id(self) -> str:
        """The unique identifier for the component."""
        return self.config["id"]

# --- Capability Mixins ---

class ProvidesDockerService(ABC):
    """Mixin for components that can be run as a Docker service."""

    @property
    def deployment(self) -> str:
        """The deployment type ('local' or 'cloud')."""
        return self.config.get("deployment", "local")
        
    @property
    @abstractmethod
    def service_name(self) -> str:
        """The name of the Docker service for this component."""
        pass

    @abstractmethod
    def get_docker_service(self) -> str:
        """Returns the Docker Compose service definition string."""
        pass

class ProvidesPythonDependencies(ABC):
    """Mixin for components that define Python-side dependencies."""

    @abstractmethod
    def get_env_vars(self) -> List[str]:
        """Returns a list of environment variable strings for the .env file."""
        pass

    @abstractmethod
    def get_requirements(self) -> List[str]:
        """Returns a list of Python package requirements."""
        pass

    @abstractmethod
    def get_imports(self) -> List[str]:
        """Returns a list of import statements required by this component."""
        pass

class ProvidesVectorDimension(ABC):
    """Mixin for components that have an associated vector dimension."""

    @abstractmethod
    def get_vector_dimension(self) -> int:
        """Returns the vector dimension."""
        pass

# --- High-Level Abstract Components ---

class EmbeddingComponent(BaseComponent, ProvidesPythonDependencies, ProvidesVectorDimension):
    """
    Abstract base class for embedding components.
    Note: Does not inherit from ProvidesDockerService by default. Concrete implementations
    that support local deployment via Docker should also inherit from ProvidesDockerService.
    """
    
    @abstractmethod
    def get_code_logic(self) -> str:
        """Returns the Python code block for the embedding logic."""
        pass

    def get_imports(self) -> List[str]:
        """Returns common imports for embedding components."""
        return [
            "import requests",
            "from typing import List, Dict, Any",
            "from config import Config",
        ]

class VectorStoreComponent(BaseComponent, ProvidesPythonDependencies):
    """
    Abstract base class for vector store components.
    Note: Does not inherit from ProvidesDockerService by default. Concrete implementations
    that support local deployment via Docker should also inherit from ProvidesDockerService.
    """
    
    @abstractmethod
    def get_init_logic(self) -> str:
        """Returns the Python code block for the __init__ method of the VectorStore class."""
        pass

    @abstractmethod
    def get_initialize_collection_logic(self) -> str:
        """Returns the Python code block for the initialize_collection method."""
        pass

    @abstractmethod
    def get_config_class(self) -> str:
        """Returns the configuration class definition for the vector store."""
        pass

    def get_imports(self) -> List[str]:
        """Returns common imports for vector store components."""
        return [
            "from typing import List, Dict, Any",
            "from pydantic import BaseModel, Field",
            "from src.config import Config",
            "from src.utils.embedder import Embedder"
        ]

class ChunkingComponent(BaseComponent, ProvidesPythonDependencies):
    """
    Abstract base class for chunking components.
    """

    @abstractmethod
    def get_code_logic(self) -> str:
        """Returns the Python code block for the chunking logic (e.g., split_text method)."""
        pass

    @abstractmethod
    def get_config_class(self) -> str:
        """Returns the configuration class definition for the chunking strategy."""
        pass

    def get_imports(self) -> List[str]:
        """Returns common imports for chunking components."""
        return [
            "from pydantic import BaseModel, Field",
            "from src.config import Config",
        ]

class RetrievalComponent(BaseComponent):
    """Base class for retrieval components."""
    
    def get_retrieval_imports(self) -> list[str]:
        """Return retrieval-specific imports."""
        return []
    
    def get_retrieval_requirements(self) -> list[str]:
        """Return retrieval-specific requirements."""
        return []
    
    def get_vectorstore_config_updates(self) -> str:
        """Return configuration updates needed for the vectorstore."""
        return ""
    
    def get_retrieval_init_logic(self, vectorstore_component) -> str:
        """Return initialization logic for this retrieval method."""
        raise NotImplementedError("Subclasses must implement get_retrieval_init_logic")
    
    def get_retrieval_method_logic(self, vectorstore_component) -> str:
        """Return the retrieval method implementation."""
        raise NotImplementedError("Subclasses must implement get_retrieval_method_logic")
    
    def supports_vectorstore(self, vectorstore_id: str) -> bool:
        """Check if this retrieval method supports the given vectorstore."""
        return True  # By default, support all vectorstores
    
    def get_search_method_name(self, vectorstore_id: str) -> str:
        """Get the appropriate search method name for the vectorstore."""
        # Default to similarity_search for all vectorstores
        return "similarity_search"