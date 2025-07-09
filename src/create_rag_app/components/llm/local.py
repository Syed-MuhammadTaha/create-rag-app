"""
Local LLM component.
"""
from textwrap import dedent
from ..base import LLMComponent

class LocalLLMComponent(LLMComponent):
    """Component for Local LLMs."""

    def get_imports(self) -> list[str]:
        return ["from langchain_openai import ChatOpenAI"]

    def get_config_class(self) -> str:
        return dedent("""
            class GeneratorConfig(BaseModel):
                model: str = Field(default=Config.LLM_GENERATION_MODEL, description="LLM model name for local model.")
                temperature: float = Field(default=Config.TEMP, description="Temperature for generation.")
                base_url: str = Field(default=Config.LOCAL_LLM_BASE_URL, description="Base URL for the local LLM.")
        """).strip()

    def get_init_logic(self) -> str:
        return dedent("""
            self.llm = ChatOpenAI(
                model=config.model,
                temperature=config.temperature,
                openai_api_key="dummy",
                openai_api_base=config.base_url
            )
        """).strip()

    def get_env_vars(self) -> list[str]:
        return [
            'LLM_GENERATION_MODEL="local-model-name"',
            'LOCAL_LLM_BASE_URL="http://localhost:11434/v1"',
            'TEMP="0.1"'
        ]

    def get_requirements(self) -> list[str]:
        return ["langchain-openai"] 