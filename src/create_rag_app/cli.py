"""
Command-line interface for RAG application creation.
"""

import questionary
import typer
from pathlib import Path
import logging
from typing import Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

from main import create_rag_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()

# Component options
VECTOR_DBS = {
    "Chroma": {
        "description": "Open-source embedding database, great for getting started",
        "supports_local": True,
        "supports_cloud": True
    },
    "Pinecone": {
        "description": "Managed vector database service, scalable",
        "supports_local": False,
        "supports_cloud": True
    }
}

LLM_PROVIDERS = {
    "cloud": {
        "OpenAI": {
            "description": "GPT-3.5/4 - Production ready",
            "endpoint": None
        },
        "HuggingFace": {
            "description": "Cloud-hosted open source models",
            "endpoint": "https://api-inference.huggingface.co"
        }
    },
    "local": {
        "Local Endpoint": {
            "description": "Your own locally deployed LLM API",
            "endpoint": "http://localhost:8000"
        }
    }
}

EMBEDDING_MODELS = {
    "Jina": {
        "description": "Top performing open source model",
        "supports_local": True,
        "supports_cloud": True
    },
    "all-MiniLM-L6-v2": {
        "description": "Fast, lightweight, good performance",
        "supports_local": True,
        "supports_cloud": False
    }
}

CHUNKING_STRATEGIES = {
    "Fixed size": {"description": "Split by character count"},
    "Semantic": {"description": "Split by semantic meaning"}
}

RETRIEVAL_METHODS = {
    "Basic Vector Search": {"description": "Simple similarity search"},
    "Hybrid Search": {"description": "Combined vector + keyword search"}
}

def format_choices(options: dict) -> list[str]:
    """Format choices with descriptions for questionary."""
    return [f"{k} - {v['description']}" for k, v in options.items()]

def extract_choice(answer: str) -> str:
    """Extract the main choice from the formatted string."""
    return answer.split(" - ")[0]

def get_deployment_preference(component_name: str, selected_option: str, options_dict: dict) -> str:
    """Get deployment preference for a component."""
    component_info = options_dict[selected_option]
    
    # If component only supports one deployment type, return that
    if component_info["supports_local"] and not component_info["supports_cloud"]:
        return "local"
    elif component_info["supports_cloud"] and not component_info["supports_local"]:
        return "cloud"
    
    # If component supports both, ask user preference
    deployment = questionary.select(
        f"\nHow would you like to use the {component_name} ({selected_option})?",
        choices=[
            "Local - Run in dockerized containers on your machine",
            "Cloud API - Use managed service"
        ]
    ).ask()
    
    return "local" if "Local" in deployment else "cloud"

def get_llm_config() -> Dict[str, Any]:
    """Get LLM configuration based on deployment preference."""
    # First ask about deployment preference
    deployment = questionary.select(
        "How would you like to use your Language Model?",
        choices=[
            "Local - Use your own deployed LLM API",
            "Cloud - Use hosted LLM services"
        ]
    ).ask()
    
    is_local = "Local" in deployment
    
    if is_local:
        # For local deployment, ask for endpoint
        endpoint = questionary.text(
            "Enter your local LLM API endpoint:",
            default=LLM_PROVIDERS["local"]["Local Endpoint"]["endpoint"]
        ).ask()
        
        return {
            "provider": "Local Endpoint",
            "deployment": "local",
            "endpoint": endpoint,
            "requires_api_key": False
        }
    else:
        # For cloud deployment, ask which provider
        provider = extract_choice(questionary.select(
            "Select your cloud LLM provider:",
            choices=format_choices(LLM_PROVIDERS["cloud"])
        ).ask())
        
        return {
            "provider": provider,
            "deployment": "cloud",
            "endpoint": LLM_PROVIDERS["cloud"][provider]["endpoint"],
            "requires_api_key": True
        }

def collect_config() -> Dict[str, Any]:
    """Collect configuration from user input."""
    console.print("\n[bold cyan]create-rag-app[/bold cyan] - RAG Application Generator\n")
    
    # Project name
    project_name = questionary.text(
        "Project name:",
        validate=lambda text: len(text) > 0,
        default="my-rag-app"
    ).ask()

    console.print("\n[bold]Components Setup[/bold]")
    
    # Vector DB
    console.print("\n[bold cyan]Vector Database[/bold cyan]")
    vector_db = extract_choice(questionary.select(
        "Select vector database:",
        choices=format_choices(VECTOR_DBS)
    ).ask())
    
    vector_db_deployment = get_deployment_preference("vector database", vector_db, VECTOR_DBS)

    # LLM Configuration
    console.print("\n[bold cyan]Language Model[/bold cyan]")
    llm_config = get_llm_config()

    # Embedding Model
    console.print("\n[bold cyan]Embedding Model[/bold cyan]")
    embedding_model = extract_choice(questionary.select(
        "Select embedding model:",
        choices=format_choices(EMBEDDING_MODELS)
    ).ask())
    
    embedding_deployment = get_deployment_preference("embedding model", embedding_model, EMBEDDING_MODELS)

    # Processing Configuration
    console.print("\n[bold cyan]Processing Configuration[/bold cyan]")
    chunking_strategy = extract_choice(questionary.select(
        "Chunking strategy:",
        choices=format_choices(CHUNKING_STRATEGIES)
    ).ask())

    retrieval_method = extract_choice(questionary.select(
        "Retrieval method:",
        choices=format_choices(RETRIEVAL_METHODS)
    ).ask())

    return {
        "project_name": project_name,
        "vector_db": {
            "provider": vector_db,
            "deployment": vector_db_deployment
        },
        "llm": llm_config,
        "embedding": {
            "model": embedding_model,
            "deployment": embedding_deployment
        },
        "chunking_strategy": chunking_strategy,
        "retrieval_method": retrieval_method
    }

def main():
    """Main CLI entrypoint."""
    try:
        console.print("\n[bold green]🚀 Welcome to create-rag-app![/bold green]")
        console.print("A modern RAG application generator\n")

        # Collect configuration
        config = collect_config()
        
        # Show summary in a panel
        console.print("\n") # Add extra space before panel
        summary = [
            "[bold]Component Deployments[/bold]",
            f"• Vector DB: [cyan]{config['vector_db']['provider']}[/cyan] ({config['vector_db']['deployment']})",
            f"• LLM: [cyan]{config['llm']['provider']}[/cyan] ({config['llm']['deployment']})",
        ]
        
        if config['llm']['endpoint']:
            summary.append(f"  └─ Endpoint: [dim]{config['llm']['endpoint']}[/dim]")
            
        summary.extend([
            f"• Embedding: [cyan]{config['embedding']['model']}[/cyan] ({config['embedding']['deployment']})",
            "",
            "[bold]Processing[/bold]",
            f"• Chunking: {config['chunking_strategy']}",
            f"• Retrieval: {config['retrieval_method']}"
        ])
        
        console.print(Panel(
            "\n".join(summary),
            title="[bold]Project Configuration[/bold]",
            expand=False
        ))
        
        # Confirm and proceed
        if questionary.confirm("\nProceed with this configuration?").ask():
            console.print("\n[bold]Creating your RAG application...[/bold]")
            
            # Generate project using main.py
            output_dir = Path.cwd()
            project_dir = create_rag_app(config, output_dir)
            
            console.print(f"\n[bold green]✨ Success![/bold green] Created {config['project_name']} at {project_dir}")
            
            # Show next steps in a panel
            console.print("\n") # Add extra space before panel
            next_steps = ["[bold]Next steps:[/bold]"]
            
            if config['llm']['deployment'] == 'local':
                next_steps.extend([
                    "",
                    "[bold cyan]LLM Setup[/bold cyan]",
                    f"• Ensure your LLM API is running at {config['llm']['endpoint']}"
                ])
            
            # Add cloud API setup instructions
            cloud_components = []
            if config['vector_db']['deployment'] == 'cloud':
                cloud_components.append(f"{config['vector_db']['provider']} (Vector DB)")
            if config['llm']['deployment'] == 'cloud':
                cloud_components.append(f"{config['llm']['provider']} (LLM)")
            if config['embedding']['deployment'] == 'cloud':
                cloud_components.append(f"{config['embedding']['model']} (Embedding)")
            
            if cloud_components:
                next_steps.extend([
                    "",
                    "[bold cyan]API Keys[/bold cyan]"
                ])
                for component in cloud_components:
                    next_steps.append(f"• Set up {component} API key in [dim].env[/dim]")
            
            # Docker setup
            next_steps.extend([
                "",
                "[bold cyan]Docker[/bold cyan]",
                "• Make sure Docker and docker-compose are installed",
                "• Run [bold]docker-compose up[/bold] to start the application"
            ])
            
            console.print(Panel(
                "\n".join(next_steps),
                title="[bold]Getting Started[/bold]",
                expand=False
            ))
            
            console.print("\nNeed help? Check out the [bold cyan]README.md[/bold cyan] for detailed instructions.")
            
        else:
            console.print("\n[dim]Operation cancelled. Run the command again to create a different configuration.[/dim]")
            
    except Exception as e:
        logger.error(f"Error creating RAG application: {str(e)}")
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        raise

if __name__ == "__main__":
    typer.run(main)