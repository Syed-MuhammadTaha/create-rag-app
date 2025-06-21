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
    def get_code_logic(self) -> str:
        """Returns the Python code block for initializing the vector store client and collection."""
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
            "from config import Config",
            "from .utils.embedder import Embedder"
        ] 