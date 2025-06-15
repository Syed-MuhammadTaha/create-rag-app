"""
Base classes for RAG application components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

class EmbeddingComponent(ABC):
    """Abstract base class for an embedding component."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the component with its specific configuration.
        Example config: {"model": "Jina", "id": "jina", "deployment": "local"}
        """
        self.config = config

    @property
    def id(self) -> str:
        """The unique identifier for the component (e.g., 'jina')."""
        return self.config["id"]

    @property
    def deployment(self) -> str:
        """The deployment type ('local' or 'cloud')."""
        return self.config["deployment"]
        
    @property
    def service_name(self) -> str:
        """The name of the Docker service for this component."""
        return f"{self.id}-embeddings"

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
    
    @abstractmethod
    def get_code_logic(self) -> str:
        """Returns the Python code block for the embedding logic."""
        pass 