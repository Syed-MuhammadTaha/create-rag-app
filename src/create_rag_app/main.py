"""
Core application logic for RAG app generation.
"""

from pathlib import Path
from typing import Dict, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape
import logging

from create_rag_app.components.embedding.jina import JinaComponent
from create_rag_app.components.embedding.all_minilm import AllMiniLMComponent
from create_rag_app.components.vectorstore.qdrant import QdrantComponent
from create_rag_app.components.chunking.fixed_size import FixedSizeChunkingComponent
from create_rag_app.components.chunking.semantic import SemanticChunkingComponent
from create_rag_app.components.retriever.dense import DenseRetrievalComponent
from create_rag_app.components.retriever.sparse import SparseRetrievalComponent
from create_rag_app.components.retriever.hybrid import HybridRetrievalComponent
from create_rag_app.components.llm.openai import OpenAIComponent
from create_rag_app.components.llm.hf import HFComponent
from create_rag_app.components.llm.local import LocalLLMComponent

logger = logging.getLogger(__name__)

EMBEDDING_REGISTRY = {
    "jina": JinaComponent,
    "all_minilm_l6_v2": AllMiniLMComponent,
}

VECTORSTORE_REGISTRY = {
    "qdrant": QdrantComponent,
}

CHUNKING_REGISTRY = {
    "fixed": FixedSizeChunkingComponent,
    "semantic": SemanticChunkingComponent,
}

RETRIEVER_REGISTRY = {
    "dense_vector_search": DenseRetrievalComponent,
    "sparse_vector_search": SparseRetrievalComponent,
    "hybrid_vector_search": HybridRetrievalComponent,
}

LLM_REGISTRY = {
    "openai": OpenAIComponent,
    "huggingface": HFComponent,
    "local": LocalLLMComponent,
}

def generate_template_context(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates the full context for Jinja2 rendering, including component instances.
    """
    context = config.copy()

    # Create embedding component instance
    embedding_id = config["embedding"]["id"]
    if embedding_id in EMBEDDING_REGISTRY:
        EmbeddingClass = EMBEDDING_REGISTRY[embedding_id]
        context["embedding_component"] = EmbeddingClass(config["embedding"])
    else:
        logger.warning(f"Embedding component '{embedding_id}' not found in registry")

        # Create retrieval component instance first (needed for vectorstore configuration)
    retriever_id = config["retriever"]["id"]
    retrieval_requirements = {}
    if retriever_id in RETRIEVER_REGISTRY:
        RetrieverClass = RETRIEVER_REGISTRY[retriever_id]
        retriever_component = RetrieverClass(config["retriever"])
        context["retriever_component"] = retriever_component
        retrieval_requirements = retriever_component.get_vectorstore_requirements()
    else:
        logger.warning(f"Retriever component '{retriever_id}' not found in registry")

    # Create vector store component instance with retrieval requirements
    vectorstore_id = config["vector_db"]["id"]
    if vectorstore_id in VECTORSTORE_REGISTRY:
        VectorStoreClass = VECTORSTORE_REGISTRY[vectorstore_id]
        context["vectorstore_component"] = VectorStoreClass(config["vector_db"], retrieval_requirements)
    else:
        logger.warning(f"VectorStore component '{vectorstore_id}' not found in registry")

    # Create chunking component instance
    chunking_id = config["chunking"]["id"]
    if chunking_id in CHUNKING_REGISTRY:
        ChunkingClass = CHUNKING_REGISTRY[chunking_id]
        context["chunking_component"] = ChunkingClass(config["chunking"])
    else:
        logger.warning(f"Chunking component '{chunking_id}' not found in registry") 

    # Create llm component instance
    llm_id = config["llm"]["id"]
    if llm_id in LLM_REGISTRY:
        LLMClass = LLM_REGISTRY[llm_id]
        context["llm_component"] = LLMClass(config["llm"])
    else:
        logger.warning(f"LLM component '{llm_id}' not found in registry")

    return context

class RAGAppGenerator:
    """Handles RAG application generation and module loading."""
    
    def __init__(self):
        self.template_dir = Path(__file__).parent.parent.parent / "templates"
        self.env = self._create_jinja_env()
        
    def _create_jinja_env(self) -> Environment:
        """Create and configure Jinja environment."""
        return Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def _render_template(self, template_path: str, context: dict) -> str:
        """Render a template with given context."""
        template = self.env.get_template(template_path)
        return template.render(**context)
    
    def generate_project(self, config: Dict[str, Any], output_dir: Path) -> Path:
        """Generate project from configuration."""
        # Create project directory
        project_dir = output_dir / config['project_name']
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create data directory and placeholder file
        data_dir = project_dir / "data"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "placeholder.pdf").touch()
        
        # Create src directory
        src_dir = project_dir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the full template context, including component instances
        context = generate_template_context(config)
        
        # Generate core files
        core_files = {
            'config.py': 'config.py.j2',
            'src/utils/embedder.py': 'src/utils/embedder.py.j2',
            'src/utils/loader.py': 'src/utils/loader.py.j2',
            'src/vectorstore.py': 'src/vectorstore.py.j2',
            'src/rag_pipeline.py': 'src/rag_pipeline.py.j2',
            'src/generator.py': 'src/generator.py.j2',
            'app.py': 'app.py.j2',
            'frontend.py': 'frontend.py.j2',
            'requirements.txt': 'requirements.txt.j2',
            'docker-compose.yml': 'docker-compose.yml.j2',
            '.env': 'env.j2'
        }
        
        for output_path, template_path in core_files.items():
            content = self._render_template(template_path, context)
            file_path = project_dir / output_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
        
        # Generate Docker files (always needed for frontend/backend)
        docker_files = {
            'Dockerfile.backend': 'Dockerfile.backend.j2',
            'Dockerfile.frontend': 'Dockerfile.frontend.j2',
        }
        
        for output_path, template_path in docker_files.items():
            content = self._render_template(template_path, context)
            file_path = project_dir / output_path
            file_path.write_text(content)
        
        return project_dir

def create_rag_app(config: Dict[str, Any], output_dir: Path) -> Path:
    """Create a new RAG application from configuration."""
    try:
        generator = RAGAppGenerator()
        return generator.generate_project(config, output_dir)
    except Exception as e:
        logger.error(f"Error creating RAG application: {str(e)}")
        raise 