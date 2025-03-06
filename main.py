import os
import yaml
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv

# 📚 Import necessary libraries

# 🚀 Load environment variables from .env file
load_dotenv()

# 📋 Define Input Model
class ResearchRequest(BaseModel):
    topic: str
    llm_name: str = None

# 🔄 Function to load LLM configurations from YAML
def load_llms():
    """Load LLM configurations from YAML file"""
    with open("config/llms.yaml", "r") as file:
        llm_configs = yaml.safe_load(file)
    
    # Process environment variables in config
    for llm_name, config in llm_configs.items():
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                config[key] = os.environ.get(env_var, "")
    
    return llm_configs

# 🤖 Initialize LLM based on configuration
def get_llm(llm_name=None):
    """Get LLM instance based on name or default"""
    llm_configs = load_llms()
    
    # If no llm_name provided, use the default
    if not llm_name:
        for name, config in llm_configs.items():
            if config.get("default", False):
                llm_name = name
                break
        # If no default is set, use the first one
        if not llm_name and llm_configs:
            llm_name = next(iter(llm_configs))
    
    # Check if the requested LLM exists
    if llm_name not in llm_configs:
        raise ValueError(f"LLM '{llm_name}' not found in configuration")
    
    # Get the LLM config
    config = llm_configs[llm_name]
    llm_type = config.get("type", "").lower()
    
    # 🔀 Initialize the proper LLM based on type
    if llm_type == "ollama":
        return LLM(
            model=config.get("model"),
            base_url=config.get("base_url")
        )
    
    elif llm_type == "gemini":
        # Extract the model name and ensure it's properly formatted
        model_name = config.get("model")
        formatted_model = f"gemini/{model_name}"
        
        # Resolve any environment variables in the API key
        api_key = config.get("api_key")
        if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]  # Extract environment variable name
            api_key = os.environ.get(env_var)
            
            if not api_key:
                raise ValueError(f"API key for Gemini not found in environment variable {env_var}")
        
        # Return configured LLM instance
        return LLM(
            model=formatted_model,
            api_key=api_key,
            temperature=config.get("temperature", 0.7)
        )
    
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")

# 🤖 Load agent configurations with topic replacement
def load_agents(topic, llm_name=None, custom_llm=None):
    """Load agent configurations and apply topic formatting"""
    with open("config/agents.yaml", "r") as file:
        agent_configs = yaml.safe_load(file)
    
    # Use provided LLM or get a new one
    llm_to_use = custom_llm or get_llm(llm_name)
    
    agents = {}
    for agent_name, config in agent_configs.items():
        # Apply topic formatting to all string fields that might contain {topic}
        role = config['role'].format(topic=topic) if '{topic}' in config.get('role', '') else config['role']
        goal = config['goal'].format(topic=topic) if '{topic}' in config.get('goal', '') else config['goal']
        backstory = config['backstory'].format(topic=topic) if '{topic}' in config.get('backstory', '') else config['backstory']
        
        agents[agent_name] = Agent(
            role=role,
            goal=goal,
            verbose=config.get('verbose', True),
            memory=config.get('memory', False),
            backstory=backstory,
            allow_delegation=config.get('allow_delegation', False),
            llm=llm_to_use
        )
    
    return agents

# 🔄 Load task configurations with topic replacement
def load_tasks(topic, agents, llm_name=None, custom_llm=None):
    """Load task configurations and apply topic formatting"""
    with open("config/tasks.yaml", "r") as file:
        task_configs = yaml.safe_load(file)
    
    # Use provided LLM or get a new one
    llm_to_use = custom_llm or get_llm(llm_name)
    
    tasks = []
    for task_name, config in task_configs.items():
        # Get the agent specified for this task
        agent_name = config['agent']
        if agent_name not in agents:
            raise ValueError(f"Agent '{agent_name}' specified in task '{task_name}' not found in available agents")
        
        # Apply topic formatting to description and expected_output
        description = config['description'].format(topic=topic) if '{topic}' in config.get('description', '') else config['description']
        expected_output = config.get('expected_output', '').format(topic=topic) if '{topic}' in config.get('expected_output', '') else config.get('expected_output', '')
        
        tasks.append(Task(
            description=description,
            expected_output=expected_output,
            llm=llm_to_use,
            agent=agents[agent_name]
        ))
    
    return tasks

# 🚀 Main function to run the crew with tasks
def run_crew_task(topic, llm_name=None):
    """Business logic to run a crew with a given topic and LLM"""
    # Get LLM instance
    llm = get_llm(llm_name)
    
    # Load agents from configuration with topic
    agents = load_agents(topic, llm_name, llm)
    
    # Load tasks from configuration with topic
    tasks = load_tasks(topic, agents, llm_name, llm)
    
    # Create the crew
    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential
    )
    
    # Execute the crew with the provided topic
    result = crew.kickoff(inputs={"topic": topic})
    return result

# 🚀 Entry point for direct script execution
if __name__ == "__main__":
    test_topic = "artificial intelligence"
    result = run_crew_task(test_topic, "gemini_remote")
    print(result)