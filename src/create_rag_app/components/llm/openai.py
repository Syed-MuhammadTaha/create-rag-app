"""
OpenAI LLM component.
"""
from textwrap import dedent
from ..base import LLMComponent

class OpenAIComponent(LLMComponent):
    """Component for OpenAI LLMs."""

    def get_imports(self) -> list[str]:
        return ["from langchain_openai import ChatOpenAI"]

    def get_config_class(self) -> str:
        return dedent("""
            class GeneratorConfig(BaseModel):
                model: str = Field(default=Config.LLM_GENERATION_MODEL, description="LLM model name for OpenAI.")
                temperature: float = Field(default=Config.TEMP, description="Temperature for generation.")
                api_key: str = Field(default=Config.OPENAI_API_KEY, description="API key for OpenAI.")
        """).strip()

    def get_init_logic(self) -> str:
        return dedent("""
            self.llm = ChatOpenAI(
                model=config.model,
                temperature=config.temperature,
                openai_api_key=config.api_key
            )
        """).strip()

    def get_env_vars(self) -> list[str]:
        return [
            'LLM_GENERATION_MODEL="gpt-4o"',
            'OPENAI_API_KEY="your-openai-api-key"',
            'TEMP="0.1"'
        ]

    def get_requirements(self) -> list[str]:
        return ["langchain-openai"] 