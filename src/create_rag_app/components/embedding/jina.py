"""
Jina embedding component.
"""
from textwrap import dedent
from create_rag_app.components.base import EmbeddingComponent

class JinaComponent(EmbeddingComponent):
    """Implementation for the Jina embedding model."""

    def get_docker_service(self) -> str:
        if self.deployment == "local":
            return dedent(f"""
            {self.service_name}:
                image: jinaai/jina-embeddings-v3:latest
                container_name: {self.service_name}
                ports:
                  - "8080:8080"
                environment:
                  - JINA_EMBEDDINGS_MODEL_NAME={self.config['model']}
                networks:
                  - app-network
            """).strip()
        return ""

    def get_env_vars(self) -> list[str]:
        if self.deployment == "cloud":
            return ['JINA_API_KEY="your-jina-api-key"', 'EMBEDDING_URL="https://api.jina.ai/v1/embeddings"']
        return ['EMBEDDING_URL="http://localhost:8080/v1/embeddings"']

    def get_requirements(self) -> list[str]:
        return []

    def get_code_logic(self) -> str:
        if self.deployment == "cloud":
            return dedent("""
                # Jina Cloud API
                headers = {
                    "Authorization": f"Bearer {Config.JINA_API_KEY}",
                    "Content-Type": "application/json"
                }
                try:
                    response = requests.post(
                        Config.EMBEDDING_URL,
                        json=data,
                        headers=headers
                    )
                    response.raise_for_status()
                    result = response.json()['data']
                except requests.exceptions.RequestException as e:
                    print(f"Error calling Jina API: {e}")
                    result = []
            """).strip()
        return dedent("""
            # Jina local server
            try:
                response = requests.post(
                    Config.EMBEDDING_URL,
                    json=data,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                result = response.json()['data']
            except requests.exceptions.RequestException as e:
                print(f"Error calling Jina local embedding server: {e}")
                result = []
        """).strip()

    def get_vector_dimension(self) -> int:
        return 384  # Jina's default dimension 