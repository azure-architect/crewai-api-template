# providers/gemini.py
from typing import Dict, Any
from crewai import LLM
from .base import BaseProvider
from .registry import ProviderRegistry

@ProviderRegistry.register("gemini")
class GeminiProvider(BaseProvider):
    """Provider for Google's Gemini LLMs"""
    
    @classmethod
    def create_llm(cls, config: Dict[str, Any]) -> LLM:
        """Create an LLM instance for Gemini"""
        # Resolve any environment variables
        config = cls.resolve_env_vars(config)
        
        # Get the configuration parameters with defaults
        model_name = config.get("model")
        api_key = config.get("api_key")
        temperature = config.get("temperature", 0.7)
        
        # Format the model name for liteLLM
        formatted_model = f"gemini/{model_name}"
        
        # Create and return the LLM
        return LLM(
            model=formatted_model,
            api_key=api_key,
            temperature=temperature
        )