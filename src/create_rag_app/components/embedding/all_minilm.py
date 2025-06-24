"""
all-MiniLM-L6-v2 embedding component.
"""
from textwrap import dedent
from ..base import EmbeddingComponent, ProvidesDockerService

class AllMiniLMComponent(EmbeddingComponent, ProvidesDockerService):
    """Implementation for the all-MiniLM-L6-v2 embedding model."""

    @property
    def service_name(self) -> str:
        """The name of the Docker service for this component."""
        return "minilm-embedder"

    def get_docker_service(self) -> str:
        # This component only supports local deployment.
        return dedent(f"""
        {self.service_name}:
            image: ghcr.io/clems4ever/torchserve-all-minilm-l6-v2:latest
            container_name: {self.service_name}
            restart: on-failure
            ports:
              - "8081:8080" # Use a different host port to avoid conflicts
            networks:
              - app-network
        """).strip()

    def get_env_vars(self) -> list[str]:
        # The URL points to the service name for Docker networking.
        return ['EMBEDDING_URL="http://minilm-embedder:8080/predictions/my_model"']

    def get_requirements(self) -> list[str]:
        return []
        
    def get_code_logic(self) -> str:
        return dedent("""
            # all-MiniLM-L6-v2 local server
            try:
                # The server expects a JSON object with an 'instances' key.
                response = requests.post(
                    Config.EMBEDDING_URL,
                    json={"instances": data},
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                # The response is a dict with a 'predictions' key containing a list of embeddings.
                result = response.json()["predictions"]
            except requests.exceptions.RequestException as e:
                print(f"Error calling MiniLM embedding server: {e}")
                result = []
        """).strip()

    def get_vector_dimension(self) -> int:
        return 384  # MiniLM's fixed dimension 