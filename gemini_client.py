import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class GeminiClient:
    """
    A simple client for Google's Gemini API that uses direct API key authentication
    """
    
    def __init__(self, api_key=None, model="gemini-1.5-flash"):
        """Initialize the Gemini client with API key and model name."""
        # First try the provided api_key, then check environment variables
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required either as parameter or environment variable")
        
        self.model = model
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        
    def generate_content(self, prompt, temperature=0.7, max_tokens=4096):
        """Generate content using the Gemini API."""
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }
        
        # Add API key as a query parameter
        url = f"{self.base_url}?key={self.api_key}"
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            result = response.json()
            
            # Extract the generated text from the response
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if parts and "text" in parts[0]:
                        return parts[0]["text"]
            
            # If we couldn't extract text using the expected structure
            return "Error: Unable to parse Gemini response"
            
        except Exception as e:
            return f"Error generating content: {str(e)}"