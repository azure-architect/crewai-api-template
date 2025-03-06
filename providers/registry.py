# providers/registry.py
from typing import Dict, Type, Any
from .base import BaseProvider

class ProviderRegistry:
    """Registry to manage and retrieve LLM providers"""
    
    _providers: Dict[str, Type[BaseProvider]] = {}
    
    @classmethod
    def register(cls, provider_type: str):
        """
        Decorator to register a provider class
        
        Args:
            provider_type: The type name used in configuration
            
        Returns:
            Decorator function
        """
        def decorator(provider_class):
            cls._providers[provider_type.lower()] = provider_class
            return provider_class
        return decorator
    
    @classmethod
    def get_provider(cls, provider_type: str) -> Type[BaseProvider]:
        """
        Get a provider class by type
        
        Args:
            provider_type: The provider type to retrieve
            
        Returns:
            The provider class or None if not found
        """
        return cls._providers.get(provider_type.lower())
    
    @classmethod
    def list_providers(cls) -> Dict[str, Type[BaseProvider]]:
        """List all registered providers"""
        return cls._providers.copy()