"""
all-MiniLM-L6-v2 embedding component.
"""
from textwrap import dedent
from create_rag_app.components.base import EmbeddingComponent

class AllMiniLMComponent(EmbeddingComponent):
    """Implementation for the all-MiniLM-L6-v2 embedding model."""

    def get_docker_service(self) -> str:
        return dedent(f"""
        {self.service_name}:
            image: ghcr.io/clems4ever/torchserve-all-minilm-l6-v2:latest
            container_name: {self.service_name}
            restart: always
            ports:
              - "8080:8080"
            depends_on:
              - qdrant
            stdin_open: true
            tty: true
            networks:
              - app-network
            expose:
              - 8080
        """).strip()

    def get_env_vars(self) -> list[str]:
        return ['EMBEDDING_URL="http://localhost:8080/predictions/all-MiniLM-L6-v2"']

    def get_requirements(self) -> list[str]:
        return ["sentence-transformers"]
        
    def get_code_logic(self) -> str:
        return dedent("""
            # all-MiniLM-L6-v2 local server
            try:
                response = requests.post(
                    Config.EMBEDDING_URL,
                    json=data,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error calling MiniLM embedding server: {e}")
                result = []
        """).strip()

    def get_vector_dimension(self) -> int:
        return 384  # MiniLM's fixed dimension 