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
                image: jinaai/jina-embeddings:0.10.0
                container_name: {self.service_name}
                ports:
                  - "5656:5656"
                environment:
                  - JINA_EMBEDDINGS_MODEL_NAME={self.config['model']}
            """).strip()
        return ""

    def get_env_vars(self) -> list[str]:
        if self.deployment == "cloud":
            return ['JINA_API_KEY="your-jina-api-key"']
        return ['JINA_EMBEDDING_URL="http://localhost:5656/embeddings"']

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
                        "https://api.jina.ai/v1/embeddings",
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
                    Config.JINA_EMBEDDING_URL,
                    json=data,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                result = response.json()['data']
            except requests.exceptions.RequestException as e:
                print(f"Error calling Jina local embedding server: {e}")
                result = []
        """).strip() 