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
            build:
              context: .
              dockerfile: Dockerfile.minilm
            container_name: {self.service_name}
            ports:
              - "8080:8080"
            command: sh -c "torch-model-archiver --model-name all-MiniLM-L6-v2 --version 1.0 --handler ./embedding_handler.py --serialized-file ./model.pt --export-path /home/model-server/model-store && torchserve --start --model-store /home/model-server/model-store --models all-MiniLM-L6-v2=all-MiniLM-L6-v2.mar"
        """).strip()

    def get_env_vars(self) -> list[str]:
        return ['MINILM_EMBEDDING_URL="http://localhost:8080/predictions/all-MiniLM-L6-v2"']

    def get_requirements(self) -> list[str]:
        return ["sentence-transformers"]
        
    def get_code_logic(self) -> str:
        return dedent("""
            # all-MiniLM-L6-v2 local server
            try:
                response = requests.post(
                    Config.MINILM_EMBEDDING_URL,
                    json=data,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error calling MiniLM embedding server: {e}")
                result = []
        """).strip() 