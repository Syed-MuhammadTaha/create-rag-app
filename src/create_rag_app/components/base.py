"""
Base classes for RAG application components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseComponent(ABC):
    """Abstract base class for all components."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the component with its specific configuration.
        """
        self.config = config

    @property
    def id(self) -> str:
        """The unique identifier for the component."""
        return self.config["id"]

    @property
    def deployment(self) -> str:
        """The deployment type ('local' or 'cloud')."""
        return self.config["deployment"]

    @abstractmethod
    def get_docker_service(self) -> str:
        """Returns the Docker Compose service definition string."""
        pass

    @abstractmethod
    def get_env_vars(self) -> List[str]:
        """Returns a list of environment variable strings for the .env file."""
        pass

    @abstractmethod
    def get_requirements(self) -> List[str]:
        """Returns a list of Python package requirements."""
        pass

class EmbeddingComponent(BaseComponent):
    """Abstract base class for embedding components."""
    
    @property
    def service_name(self) -> str:
        """The name of the Docker service for this component."""
        return f"{self.id}-embedder"
    
    @abstractmethod
    def get_code_logic(self) -> str:
        """Returns the Python code block for the embedding logic."""
        pass

    @abstractmethod
    def get_vector_dimension(self) -> int:
        """Returns the vector dimension for this embedding model."""
        pass

class VectorStoreComponent(BaseComponent):
    """Abstract base class for vector store components."""
    
    @property
    def service_name(self) -> str:
        """The name of the Docker service for this component."""
        return f"{self.id}-vectorstore"

    @abstractmethod
    def get_collection_init_logic(self) -> str:
        """Returns the Python code block for initializing the vector store collection."""
        pass

    @abstractmethod
    def get_config_class(self) -> str:
        """Returns the configuration class definition for the vector store."""
        pass

    @abstractmethod
    def get_vector_dimension(self) -> int:
        """Returns the vector dimension required for the store."""
        pass 