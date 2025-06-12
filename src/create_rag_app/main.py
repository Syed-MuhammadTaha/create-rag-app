"""
Core application logic for RAG app generation.
"""

from pathlib import Path
from typing import Dict, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape
import logging

logger = logging.getLogger(__name__)

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
    
    def _load_component(self, component_type: str, choice: str) -> str:
        """Load a specific component template."""
        template_path = f"components/{component_type}/{choice.lower()}.j2"
        try:
            return self.env.get_template(template_path).render()
        except Exception as e:
            logger.error(f"Error loading component {template_path}: {str(e)}")
            raise
    
    def generate_project(self, config: Dict[str, Any], output_dir: Path) -> Path:
        """Generate project from configuration."""
        # Create project directory
        project_dir = output_dir / config['project_name']
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create src directory
        src_dir = project_dir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        
        # Load selected components
        components = {
            'vectorstore': self._load_component('vectorstore', config['vector_db']['provider']),
            'llm': self._load_component('llm', config['llm']['provider']),
            'embedding': self._load_component('embedding', config['embedding']['model'].split('/')[0]),
            'chunking': self._load_component('chunking', config['chunking_strategy'].lower().replace(' ', '_')),
            'retrieval': self._load_component('retrieval', config['retrieval_method'].lower().replace(' ', '_'))
        }
        
        # Update config with component implementations
        config['components'] = components
        
        # Generate core files
        core_files = {
            'src/config.py': 'base/src/config.py.j2',
            'src/rag_pipeline.py': 'base/src/rag_pipeline.py.j2',
            'src/generator.py': 'base/src/generator.py.j2',
            'src/vectorstore.py': 'base/src/vectorstore.py.j2',
            # 'src/utils/embedder.py': 'base/src/utils/embedder.py.j2',
            # 'src/utils/loader.py': 'base/src/utils/loader.py.j2',
            # 'src/utils/pydantic.py': 'base/src/utils/pydantic.py.j2',
            'app.py': 'base/app.py.j2',
            'frontend.py': 'base/frontend.py.j2',
            'requirements.txt': 'base/requirements.txt.j2',
            '.env': 'base/env.j2'
        }
        
        for output_path, template_path in core_files.items():
            content = self._render_template(template_path, config)
            file_path = project_dir / output_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
        
        # Generate Docker files (always needed for frontend/backend)
        docker_files = {
            'Dockerfile.backend': 'base/Dockerfile.backend.j2',
            'Dockerfile.frontend': 'base/Dockerfile.frontend.j2',
            'docker-compose.yml': 'base/docker-compose.yml.j2'
        }
        
        for output_path, template_path in docker_files.items():
            content = self._render_template(template_path, config)
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