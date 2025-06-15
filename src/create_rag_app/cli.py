"""
Command-line interface for RAG application creation.
"""

import sys
from pathlib import Path
import os
import questionary
import typer
import logging
from typing import Dict, Any
from rich.console import Console
from rich.panel import Panel
from create_rag_app.main import create_rag_app

# This ensures the application can be run as a script
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


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

LLM_OPTIONS = {
    "OpenAI": {
        "description": "GPT-3.5/4 - Production ready",
        "type": "cloud",
        "requires_api_key": True
    },
    "HuggingFace": {
        "description": "Cloud-hosted open source models",
        "type": "cloud",
        "requires_api_key": True
    },
    "Local": {
        "description": "Your own locally deployed LLM API",
        "type": "local",
        "requires_api_key": False
    }
}

EMBEDDING_MODELS = {
    "Jina": {
        "description": "Top performing open source model",
        "supports_local": True,
        "supports_cloud": True,
        "id": "jina"
    },
    "all-MiniLM-L6-v2": {
        "description": "Fast, lightweight, good performance",
        "supports_local": True,
        "supports_cloud": False,
        "id": "all_minilm_l6_v2"
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
    """Get LLM configuration."""
    # Ask which LLM to use
    llm_choice = extract_choice(questionary.select(
        "Select your Language Model:",
        choices=format_choices(LLM_OPTIONS)
    ).ask())
    
    llm_info = LLM_OPTIONS[llm_choice]
    return {
        "model": llm_choice,
        "type": llm_info["type"],
        "requires_api_key": llm_info["requires_api_key"]
    }

def collect_config() -> Dict[str, Any]:
    """Collect configuration from user input."""
    
    # Project name
    console.print("[bold yellow]PROJECT SETUP[/bold yellow] 📝")
    project_name = questionary.text(
        "Project name:",
        validate=lambda text: len(text) > 0,
        default="my-rag-app"
    ).ask()

    console.print("\n[bold green]COMPONENTS SETUP[/bold green] 🛠️\n")
    
    # 1. Embedding Model (First because it's fundamental to RAG)
    console.print("[bold blue]1. Embedding Model[/bold blue] 🧬")
    console.print("[dim]Choose how to convert text to vectors[/dim]")
    embedding_model_choice = extract_choice(questionary.select(
        "Select embedding model:",
        choices=format_choices(EMBEDDING_MODELS)
    ).ask())
    
    embedding_deployment = get_deployment_preference("embedding model", embedding_model_choice, EMBEDDING_MODELS)

    # 2. Vector DB (Second because it stores the embeddings)
    console.print("\n[bold purple]2. Vector Database[/bold purple] 🗄️")
    console.print("[dim]Select where to store your vectors[/dim]")
    vector_db = extract_choice(questionary.select(
        "Select vector database:",
        choices=format_choices(VECTOR_DBS)
    ).ask())
    
    vector_db_deployment = get_deployment_preference("vector database", vector_db, VECTOR_DBS)

    # 3. Processing Configuration (Third because it affects how documents are processed)
    console.print("\n[bold orange1]3. Document Processing[/bold orange1] 📄")
    console.print("[dim]Configure how your documents are split[/dim]")
    chunking_strategy = extract_choice(questionary.select(
        "How should documents be split into chunks?",
        choices=format_choices(CHUNKING_STRATEGIES)
    ).ask())

    # 4. Retrieval Method (Fourth because it depends on chunking and affects how documents are retrieved)
    console.print("\n[bold red]4. Retrieval Strategy[/bold red] 🔍")
    console.print("[dim]Define how to find relevant information[/dim]")
    retrieval_method = extract_choice(questionary.select(
        "How should relevant chunks be retrieved?",
        choices=format_choices(RETRIEVAL_METHODS)
    ).ask())

    # 5. LLM Configuration (Last because it's the final step in the RAG pipeline)
    console.print("\n[bold green1]5. Language Model[/bold green1] 🤖")
    console.print("[dim]Choose your AI model for generating responses[/dim]")
    llm_config = get_llm_config()

    return {
        "project_name": project_name,
        "embedding": {
            "model": embedding_model_choice,
            "id": EMBEDDING_MODELS[embedding_model_choice]["id"],
            "deployment": embedding_deployment
        },
        "vector_db": {
            "provider": vector_db,
            "deployment": vector_db_deployment
        },
        "chunking_strategy": chunking_strategy,
        "retrieval_method": retrieval_method,
        "llm": llm_config
    }

def main():
    """Main CLI entrypoint."""
    try:
        console.print("\n[bold magenta]create-rag-app[/bold magenta] 🚀")
        console.print("[dim italic]A modern RAG application generator[/dim italic]\n")

        # Collect configuration
        config = collect_config()
        
        # Show summary in a panel
        console.print("\n") # Add extra space before panel
        summary = [
            "[bold green]✨ Your RAG App Configuration[/bold green]",
            "",
            "[bold blue]Embedding[/bold blue] 🧬",
            f"• Model: [cyan]{config['embedding']['model']}[/cyan]",
            f"• Deployment: [cyan]{config['embedding']['deployment']}[/cyan]",
            "",
            "[bold purple]Vector Database[/bold purple] 🗄️",
            f"• Provider: [cyan]{config['vector_db']['provider']}[/cyan]",
            f"• Deployment: [cyan]{config['vector_db']['deployment']}[/cyan]",
            "",
            "[bold orange1]Processing[/bold orange1] 📄",
            f"• Chunking: [cyan]{config['chunking_strategy']}[/cyan]",
            "",
            "[bold red]Retrieval[/bold red] 🔍",
            f"• Method: [cyan]{config['retrieval_method']}[/cyan]",
            "",
            "[bold green1]Language Model[/bold green1] 🤖",
            f"• Provider: [cyan]{config['llm']['model']}[/cyan]",
            f"• Type: [cyan]{config['llm']['type']}[/cyan]"
        ]
        
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
            
            if config['llm']['type'] == 'local':
                next_steps.extend([
                    "",
                    "[bold cyan]LLM Setup[/bold cyan]",
                    "• Configure your local LLM API endpoint in [dim].env[/dim]"
                ])
            
            # Add cloud API setup instructions
            cloud_components = []
            if config['vector_db']['deployment'] == 'cloud':
                cloud_components.append(f"{config['vector_db']['provider']} (Vector DB)")
            if config['llm']['type'] == 'cloud':
                cloud_components.append(f"{config['llm']['model']} (LLM)")
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