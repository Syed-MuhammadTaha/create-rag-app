import questionary
import typer
from typing import Optional
import json
from pathlib import Path

# Define options for different components
VECTOR_DBS = {
    "Chroma": {"description": "Open-source embedding database, great for getting started", "supports_docker": True},
    "Weaviate": {"description": "Production-ready vector database with rich features", "supports_docker": True},
    "Pinecone": {"description": "Managed vector database service, scalable", "supports_docker": False},
    "Milvus": {"description": "Open-source vector database, highly scalable", "supports_docker": True}
}

CLOUD_LLM_PROVIDERS = {
    "OpenAI": {"description": "GPT-3.5/4 - Powerful, reliable, cost-effective"},
    "LLama": {"description": "Fast Inference Cloud APIs"},
    "HuggingFace": {"description": "Open Source Models"},
    "Mistral": {"description": "Open Source Models"}
}

LOCAL_LLM_PROVIDERS = {
    "Local API": {"description": "Your own locally deployed LLM API", "endpoint_default": "http://localhost:8000"}
}

EMBEDDING_MODELS = {
    "OpenAI Ada 002": {"description": "Strong performance, widely used", "supports_docker": False},
    "BAAI/bge-large-en": {"description": "Top performing open source model", "supports_docker": True},
    "Instructor-XL": {"description": "Task-specific embeddings, open source", "supports_docker": True},
    "all-MiniLM-L6-v2": {"description": "Fast, lightweight, good performance", "supports_docker": True}
}

CHUNKING_STRATEGIES = {
    "Fixed size": {"description": "Split by character count"},
    "Paragraph": {"description": "Split by paragraphs"},
    "Semantic": {"description": "Split by semantic meaning"},
    "Hybrid": {"description": "Combine multiple strategies"}
}

RETRIEVAL_METHODS = {
    "Basic Vector Search": {"description": "Simple similarity search"},
    "Hybrid Search": {"description": "Combined vector + keyword search"},
    "Re-ranking": {"description": "Two-stage retrieval with cross-encoder"},
    "Multi-query": {"description": "Generate multiple queries for better recall"}
}

def format_choices(options: dict) -> list[str]:
    """Format choices with descriptions for questionary"""
    return [f"{k} - {v['description']}" for k, v in options.items()]

def extract_choice(answer: str) -> str:
    """Extract the main choice from the formatted string"""
    return answer.split(" - ")[0]

def get_deployment_preference(component: str, selected_option: str, options_dict: dict) -> str:
    """Get deployment preference for a component if it supports both cloud and docker"""
    if not options_dict[selected_option]["supports_docker"]:
        return "cloud"
    
    return questionary.select(
        f"How would you like to deploy the {component}?",
        choices=[
            "Docker - Run locally in containers",
            "Cloud API - Use managed service"
        ]
    ).ask().split(" - ")[0]

def get_llm_configuration():
    """Get LLM configuration based on deployment type"""
    # First ask if using cloud or local LLM
    deployment_type = questionary.select(
        "Are you using a cloud LLM provider or a locally deployed LLM?",
        choices=[
            "Cloud Provider - Use hosted LLM services (OpenAI, Anthropic, etc.)",
            "Local Deployment - Connect to your own deployed LLM API"
        ]
    ).ask().split(" - ")[0]

    if deployment_type == "Cloud Provider":
        # Ask which cloud provider
        provider = extract_choice(questionary.select(
            "Select your cloud LLM provider:",
            choices=format_choices(CLOUD_LLM_PROVIDERS)
        ).ask())
        
        return {
            "deployment": "cloud",
            "provider": provider,
            "requires_api_key": True
        }
    else:
        # Get local LLM endpoint
        endpoint = questionary.text(
            "Enter your local LLM API endpoint:",
            default="http://localhost:8000"
        ).ask()
        
        return {
            "deployment": "local",
            "endpoint": endpoint,
            "requires_api_key": False
        }

def create_app_config():
    """Gather all configuration options from the user"""
    
    # Project name
    project_name = questionary.text(
        "What is your RAG application name?",
        validate=lambda text: len(text) > 0,
        default="my-rag-app"
    ).ask()

    # Vector DB
    vector_db = extract_choice(questionary.select(
        "Select your vector database:",
        choices=format_choices(VECTOR_DBS),
    ).ask())
    
    # Vector DB deployment
    vector_db_deployment = get_deployment_preference("vector database", vector_db, VECTOR_DBS)

    # LLM Configuration
    llm_config = get_llm_configuration()

    # Embedding Model
    embedding_model = extract_choice(questionary.select(
        "Select your embedding model:",
        choices=format_choices(EMBEDDING_MODELS),
    ).ask())
    
    # Embedding deployment
    embedding_deployment = get_deployment_preference("embedding model", embedding_model, EMBEDDING_MODELS)

    # Chunking Strategy
    chunking_strategy = extract_choice(questionary.select(
        "Select your text chunking strategy:",
        choices=format_choices(CHUNKING_STRATEGIES),
    ).ask())

    # Retrieval Method
    retrieval_method = extract_choice(questionary.select(
        "Select your retrieval method:",
        choices=format_choices(RETRIEVAL_METHODS),
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
    """Main CLI entrypoint"""
    print("🚀 Welcome to Create RAG App! Let's set up your project.")
    print("Please answer a few questions to configure your RAG application.\n")

    config = create_app_config()
    
    # Show summary
    print("\n📋 Your RAG App Configuration:")
    print(json.dumps(config, indent=2))
    
    # Confirm and proceed
    if questionary.confirm("Would you like to proceed with this configuration?").ask():
        print("\n⚙️ Creating your RAG application...")
        # TODO: Add template generation logic here
        print(f"✨ Successfully created {config['project_name']}!")
        
        # Show next steps based on deployment choices
        print("\n📝 Next steps:")
        if config["llm"]["deployment"] == "local":
            print(f"- Ensure your local LLM API is running at {config['llm']['endpoint']}")
        else:
            print(f"- Set up your {config['llm']['provider']} API key in the .env file")
        
        if "Docker" in [config["vector_db"]["deployment"], config["embedding"]["deployment"]]:
            print("- Make sure Docker is installed and running")
            print("- Run `docker-compose up` to start the services")
        
        if "Cloud" in [config["vector_db"]["deployment"], config["embedding"]["deployment"]]:
            print("- Set up your cloud API keys in the .env file")
    else:
        print("\n🔄 Feel free to run the command again to create a different configuration.")

if __name__ == "__main__":
    typer.run(main)