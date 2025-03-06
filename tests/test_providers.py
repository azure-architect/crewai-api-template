import os
import sys
import yaml
from dotenv import load_dotenv

# Add the project root directory to the Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the provider system
from providers import create_llm_from_config

# Load environment variables
load_dotenv()

def load_llms():
    """Load LLM configurations from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'llms.yaml')
    with open(config_path, "r") as file:
        llm_configs = yaml.safe_load(file)
    return llm_configs

def test_provider_system():
    """Test the provider system with different LLM configurations"""
    llm_configs = load_llms()
    
    print("Testing LLM provider system...")
    results = []
    
    # Test each configured LLM
    for llm_name, config in llm_configs.items():
        print(f"\nTesting '{llm_name}' configuration:")
        print(f"  Type: {config.get('type')}")
        print(f"  Model: {config.get('model')}")
        
        try:
            # Create the LLM instance
            llm = create_llm_from_config(config)
            print(f"✅ Successfully created LLM instance: {llm.__class__.__name__}")
            print(f"  Model: {llm.model}")
            
            # Print additional parameters based on provider type
            if config.get("type") == "ollama":
                print(f"  Base URL: {llm.base_url}")
            elif config.get("type") == "gemini":
                print(f"  Temperature: {llm.temperature}")
            
            results.append((llm_name, True))
        except Exception as e:
            print(f"❌ Error creating LLM: {str(e)}")
            results.append((llm_name, False, str(e)))
    
    # Print summary
    print("\n=== Test Summary ===")
    passed = sum(1 for r in results if r[1])
    total = len(results)
    print(f"Passed: {passed}/{total} tests")
    
    return results

if __name__ == "__main__":
    test_provider_system()