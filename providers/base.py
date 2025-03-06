# providers/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from crewai import LLM

class BaseProvider(ABC):
    """Base abstract class that all LLM providers must implement"""
    
    @classmethod
    @abstractmethod
    def create_llm(cls, config: Dict[str, Any]) -> LLM:
        """
        Create and return an LLM instance based on the provider configuration
        
        Args:
            config: The provider-specific configuration
            
        Returns:
            LLM: A configured LLM instance
        """
        pass
    
    @classmethod
    def resolve_env_vars(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolves environment variables in configuration values
        
        Args:
            config: The provider configuration with potential env vars
            
        Returns:
            Dict with resolved environment variables
        """
        import os
        resolved_config = config.copy()
        
        for key, value in resolved_config.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                env_value = os.environ.get(env_var)
                if not env_value:
                    raise ValueError(f"Environment variable '{env_var}' not found")
                resolved_config[key] = env_value
                
        return resolved_config