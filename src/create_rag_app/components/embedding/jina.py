"""
Jina embedding component.
"""
from textwrap import dedent
from ..base import EmbeddingComponent, ProvidesDockerService

class JinaComponent(EmbeddingComponent, ProvidesDockerService):
    """Implementation for the Jina embedding model, supporting both cloud and local deployments."""

    @property
    def service_name(self) -> str:
        """The name of the Docker service for this component."""
        return f"{self.id}-embedder"

    def get_docker_service(self) -> str:
        if self.config.get("deployment") == "local":
            return dedent(f"""
            {self.service_name}:
                image: jinaai/jina-embeddings-v2-base-en:latest
                container_name: {self.service_name}
                ports:
                  - "8080:8080"
                networks:
                  - app-network
            """).strip()
        return ""

    def get_env_vars(self) -> list[str]:
        if self.config.get("deployment") == "cloud":
            return [
                'JINA_API_KEY="your-jina-api-key"', 
                'EMBEDDING_URL="https://api.jina.ai/v1/embeddings"', 
                'EMBEDDING_MODEL="jina-embeddings-v2-base-en"'
            ]
        # For local deployment
        return ['EMBEDDING_URL="http://jina-embedder:8080/v1/embeddings"']

    def get_requirements(self) -> list[str]:
        # No special requirements beyond the base 'requests'
        return []

    def get_code_logic(self) -> str:
        if self.config.get("deployment") == "cloud":
            return dedent("""
                # Jina Cloud API
                headers = {
                    "Authorization": f"Bearer {Config.JINA_API_KEY}",
                    "Content-Type": "application/json"
                }
                try:
                    response = requests.post(
                        Config.EMBEDDING_URL,
                        json={
                            "input": data,
                            "model": Config.EMBEDDING_MODEL
                        },
                        headers=headers
                    )
                    response.raise_for_status()
                    result = response.json()['data'][0]['embedding']
                except requests.exceptions.RequestException as e:
                    print(f"Error calling Jina API: {e}")
                    result = []
            """).strip()
        
        # Jina local server
        return dedent("""
            try:
                response = requests.post(
                    Config.EMBEDDING_URL,
                    json={"input": data},
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                # The local model returns a list of embeddings, take the first one
                result = response.json()["data"][0]["embedding"]
            except requests.exceptions.RequestException as e:
                print(f"Error calling Jina local embedding server: {e}")
                result = []
        """).strip()

    def get_vector_dimension(self) -> int:
        # jina-embeddings-v2-base-en has a dimension of 768
        return 768 