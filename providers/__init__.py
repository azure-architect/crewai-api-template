# providers/__init__.py
from .registry import ProviderRegistry
from .base import BaseProvider
# Import all providers to ensure they're registered
from . import ollama, gemini

# Add more providers here as you create them
# from . import anthropic, cohere, etc.

def create_llm_from_config(config):
    """
    Create an LLM instance from a configuration dictionary
    
    Args:
        config: Provider configuration
        
    Returns:
        An LLM instance
    """
    provider_type = config.get("type", "").lower()
    provider_class = ProviderRegistry.get_provider(provider_type)
    
    if not provider_class:
        raise ValueError(f"Unsupported LLM provider type: {provider_type}")
    
    return provider_class.create_llm(config)

__all__ = ["ProviderRegistry", "BaseProvider", "create_llm_from_config"]