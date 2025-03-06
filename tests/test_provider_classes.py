import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the project root directory to the Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the provider components
from providers import BaseProvider, ProviderRegistry
from providers.ollama import OllamaProvider
from providers.gemini import GeminiProvider


class TestProviderRegistry(unittest.TestCase):
    """Test the ProviderRegistry functionality"""
    
    def test_registry_contains_providers(self):
        """Test that the registry contains our registered providers"""
        providers = ProviderRegistry._providers
        self.assertIn('ollama', providers)
        self.assertIn('gemini', providers)
    
    def test_get_provider(self):
        """Test retrieving providers from the registry"""
        ollama = ProviderRegistry.get_provider('ollama')
        gemini = ProviderRegistry.get_provider('gemini')
        
        self.assertEqual(ollama, OllamaProvider)
        self.assertEqual(gemini, GeminiProvider)
    
    def test_case_insensitivity(self):
        """Test that provider retrieval is case insensitive"""
        ollama_upper = ProviderRegistry.get_provider('OLLAMA')
        gemini_mixed = ProviderRegistry.get_provider('GeMiNi')
        
        self.assertEqual(ollama_upper, OllamaProvider)
        self.assertEqual(gemini_mixed, GeminiProvider)


class TestOllamaProvider(unittest.TestCase):
    """Test the OllamaProvider functionality"""
    
    def test_resolve_env_vars(self):
        """Test environment variable resolution in config"""
        # Set up a test environment variable
        os.environ['TEST_BASE_URL'] = 'http://test.ollama.local:11434'
        
        config = {
            'model': 'llama2',
            'base_url': '${TEST_BASE_URL}'
        }
        
        resolved = OllamaProvider.resolve_env_vars(config)
        self.assertEqual(resolved['base_url'], 'http://test.ollama.local:11434')
    
    @patch('crewai.LLM')
    def test_create_llm(self, mock_llm):
        """Test creating an LLM instance"""
        # Set up mock
        mock_instance = MagicMock()
        mock_llm.return_value = mock_instance
        
        config = {
            'model': 'llama2',
            'base_url': 'http://localhost:11434'
        }
        
        llm = OllamaProvider.create_llm(config)
        
        # Assert LLM was created with correct parameters
        mock_llm.assert_called_once_with(
            model='llama2',
            base_url='http://localhost:11434'
        )
        
        # Assert the returned object is our mock
        self.assertEqual(llm, mock_instance)


class TestGeminiProvider(unittest.TestCase):
    """Test the GeminiProvider functionality"""
    
    def test_resolve_env_vars(self):
        """Test environment variable resolution in config"""
        # Set up a test environment variable
        os.environ['TEST_API_KEY'] = 'test-api-key-12345'
        
        config = {
            'model': 'gemini-1.5-pro',
            'api_key': '${TEST_API_KEY}',
            'temperature': 0.7
        }
        
        resolved = GeminiProvider.resolve_env_vars(config)
        self.assertEqual(resolved['api_key'], 'test-api-key-12345')
    
    @patch('crewai.LLM')
    def test_create_llm(self, mock_llm):
        """Test creating an LLM instance"""
        # Set up mock
        mock_instance = MagicMock()
        mock_llm.return_value = mock_instance
        
        config = {
            'model': 'gemini-1.5-pro',
            'api_key': 'test-api-key',
            'temperature': 0.7
        }
        
        llm = GeminiProvider.create_llm(config)
        
        # Assert LLM was created with correct parameters
        mock_llm.assert_called_once_with(
            model='gemini/gemini-1.5-pro',
            api_key='test-api-key',
            temperature=0.7
        )
        
        # Assert the returned object is our mock
        self.assertEqual(llm, mock_instance)
    
    @patch('crewai.LLM')
    def test_default_temperature(self, mock_llm):
        """Test default temperature is used when not provided"""
        # Set up mock
        mock_instance = MagicMock()
        mock_llm.return_value = mock_instance
        
        config = {
            'model': 'gemini-1.5-pro',
            'api_key': 'test-api-key'
        }
        
        llm = GeminiProvider.create_llm(config)
        
        # Assert LLM was created with default temperature
        mock_llm.assert_called_once_with(
            model='gemini/gemini-1.5-pro',
            api_key='test-api-key',
            temperature=0.7
        )


if __name__ == '__main__':
    unittest.main()