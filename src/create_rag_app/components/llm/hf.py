"""
Hugging Face LLM component.
"""
from textwrap import dedent
from ..base import LLMComponent

class HFComponent(LLMComponent):
    """Component for Hugging Face LLMs."""

    def get_imports(self) -> list[str]:
        return ["from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint"]

    def get_config_class(self) -> str:
        return dedent("""
            class GeneratorConfig(BaseModel):
                repo_id: str = Field(default=Config.LLM_GENERATION_MODEL, description="Hugging Face model repository ID.")
                task: str = Field(default=Config.HF_TASK, description="Task for the Hugging Face model.")
                max_new_tokens: int = Field(default=Config.HF_MAX_NEW_TOKENS, description="Max new tokens for generation.")
                temperature: float = Field(default=Config.TEMP, description="Temperature for generation.")
                api_key: str = Field(default=Config.HUGGINGFACEHUB_API_TOKEN, description="API key for Hugging Face.")
        """).strip()

    def get_init_logic(self) -> str:
        return dedent("""
            llm = HuggingFaceEndpoint(
                repo_id=config.repo_id,
                task=config.task,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.03,
                huggingfacehub_api_token=config.api_key
            )
            self.llm = ChatHuggingFace(llm=llm)
        """).strip()

    def get_env_vars(self) -> list[str]:
        return [
            'LLM_GENERATION_MODEL="meta-llama/Llama-3.1-8B-Instruct"',
            'HUGGINGFACEHUB_API_TOKEN="your-huggingface-api-key"',
            'HF_TASK="text-generation"',
            'HF_MAX_NEW_TOKENS="512"',
            'TEMP="0.1"'
        ]

    def get_requirements(self) -> list[str]:
        return ["langchain-huggingface", "huggingface_hub"] 