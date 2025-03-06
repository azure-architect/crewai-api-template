"""
Custom Gemini LLM implementation for CrewAI
Uses direct API key authentication instead of service account credentials
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional

# For CrewAI 0.102.0, we need to use the LangChain LLM base classes
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

class CustomGeminiLLM(LLM):
    """
    CrewAI LLM implementation that uses Google's Gemini API with
    direct API key authentication.
    """
    
    api_key: Optional[str] = None
    model: str = "gemini-1.5-pro-latest"
    temperature: float = 0.7
    max_tokens: int = 8192
    top_p: float = 0.95
    timeout: int = 120
    api_base: str = "https://generativelanguage.googleapis.com/v1beta/models"
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "gemini"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-pro-latest",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        top_p: float = 0.95,
        timeout: int = 120,
        api_base: str = "https://generativelanguage.googleapis.com/v1beta/models",
        **kwargs
    ):
        """
        Initialize the Gemini LLM with API key authentication.
        
        Args:
            api_key: Gemini API key (will use GEMINI_API_KEY env var if not provided)
            model: Gemini model name
            temperature: Temperature for response generation
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            timeout: Request timeout in seconds
            api_base: Base URL for the Gemini API
        """
        super().__init__(**kwargs)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Either pass it as api_key or set the GEMINI_API_KEY environment variable."
            )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.api_base = api_base

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """
        Make a direct API call to Gemini.
        
        Args:
            prompt: The prompt to send to Gemini
            stop: Optional list of stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated response text
        """
        # Build the API URL with model and API key
        url = f"{self.api_base}/{self.model}:generateContent?key={self.api_key}"
        
        # Override parameters from kwargs if provided
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        
        # Prepare the request payload
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": top_p
            }
        }
        
        # Add stop sequences if provided
        if stop:
            payload["generationConfig"]["stopSequences"] = stop
        
        # Make the API request
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse the response JSON
            response_json = response.json()
            
            # Extract and return the generated text
            if "candidates" in response_json and len(response_json["candidates"]) > 0:
                candidate = response_json["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if len(parts) > 0 and "text" in parts[0]:
                        return parts[0]["text"]
            
            # If we couldn't extract text in the expected format
            raise ValueError(f"Unexpected response format: {response_json}")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling Gemini API: {str(e)}")