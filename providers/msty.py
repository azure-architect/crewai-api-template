from typing import Dict, Any
from crewai import LLM
from .base import BaseProvider
from .registry import ProviderRegistry

@ProviderRegistry.register("openai")
class OpenAIProvider(BaseProvider):
    """Provider for OpenAI and OpenAI-compatible LLMs"""
    
    @classmethod
    def create_llm(cls, config: Dict[str, Any]) -> LLM:
        """Create an LLM instance for OpenAI or OpenAI-compatible endpoints"""
        # Resolve any environment variables
        config = cls.resolve_env_vars(config)
        
        # Get the configuration parameters with defaults
        model = config.get("model")
        api_base = config.get("api_base")
        api_key = config.get("api_key")
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", None)
        
        # Create and return the LLM
        return LLM(
            provider="openai",  # Specify the provider
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )