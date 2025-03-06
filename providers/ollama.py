# providers/ollama.py
from typing import Dict, Any
from crewai import LLM
from .base import BaseProvider
from .registry import ProviderRegistry

@ProviderRegistry.register("ollama")
class OllamaProvider(BaseProvider):
    """Provider for Ollama LLMs"""
    
    @classmethod
    def create_llm(cls, config: Dict[str, Any]) -> LLM:
        """Create an LLM instance for Ollama"""
        # Resolve any environment variables
        config = cls.resolve_env_vars(config)
        
        # Get the configuration parameters with defaults
        model = config.get("model")
        base_url = config.get("base_url")
        
        # Create and return the LLM
        return LLM(
            model=model,
            base_url=base_url
        )