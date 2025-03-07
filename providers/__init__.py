# providers/__init__.py
from .registry import ProviderRegistry
from .base import BaseProvider
# Import all providers to ensure they're registered
from . import ollama, gemini, msty  # Make sure msty is imported!



def create_llm_from_config(config):

    provider_type = config.get("type", "").lower()
    provider_class = ProviderRegistry.get_provider(provider_type)
    
    if not provider_class:
        raise ValueError(f"Unsupported LLM provider type: {provider_type}")
    
    return provider_class.create_llm(config)

__all__ = ["ProviderRegistry", "BaseProvider", "create_llm_from_config"]