import os
import yaml
from pydantic import BaseModel
# 📚 Import core CrewAI components
from crewai import Agent, Task, Crew, Process, LLM
# 📚 Load environment variables support
from dotenv import load_dotenv

# 🔄 Load environment variables at startup
load_dotenv()

# 📋 Define Input Model for research requests
class ResearchRequest(BaseModel):
    topic: str
    llm_name: str = None  # Optional parameter for LLM selection

# 🔄 Function to load LLM configurations from YAML
def load_llms():
    """Load LLM configurations from YAML file"""
    with open("config/llms.yaml", "r") as file:
        llm_configs = yaml.safe_load(file)
    
    # ⚙️ Process environment variables in config
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
    
    # 🔀 Default LLM selection logic
    if not llm_name:
        for name, config in llm_configs.items():
            if config.get("default", False):
                llm_name = name
                break
        # If no default is set, use the first one
        if not llm_name and llm_configs:
            llm_name = next(iter(llm_configs))
    
    # ⚠️ Check if the requested LLM exists
    if llm_name not in llm_configs:
        raise ValueError(f"LLM '{llm_name}' not found in configuration")
    
    # ⚙️ Get the LLM config
    config = llm_configs[llm_name]
    llm_type = config.get("type", "").lower()
    
    # 🤖 Initialize LLM based on type
    if llm_type == "ollama":
        return LLM(
            model=config.get("model"),
            base_url=config.get("base_url")
        )
    
    elif llm_type == "gemini":
        # 🔄 Extract the model name
        model_name = config.get("model")
        
        # 🔒 Resolve any environment variables in the API key
        api_key = config.get("api_key")
        if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var)
            
            # ⚠️ Error if API key is missing
            if not api_key:
                raise ValueError(f"API key for Gemini not found in environment variable {env_var}")
        
        # 🤖 Create Gemini LLM instance with proper provider settings
        return LLM(
            model=model_name,  # Use model name directly without prefix
            api_key=api_key,
            provider="google",  # Specify provider explicitly
            temperature=config.get("temperature", 0.7)
        )
    else:
        # ⚠️ Handle unsupported LLM types
        raise ValueError(f"Unsupported LLM type: {llm_type}")

# 🔄 Load agent configurations and apply topic formatting
def load_agents(topic, llm_name=None, custom_llm=None):
    """Load agent configurations and apply topic formatting"""
    with open("config/agents.yaml", "r") as file:
        agent_configs = yaml.safe_load(file)
    
    # 🤖 Use provided LLM or get a new one
    llm_to_use = custom_llm or get_llm(llm_name)
    
    agents = {}
    for agent_name, config in agent_configs.items():
        # 🔍 Apply topic formatting to string fields
        role = config['role'].format(topic=topic) if '{topic}' in config.get('role', '') else config['role']
        goal = config['goal'].format(topic=topic) if '{topic}' in config.get('goal', '') else config['goal']
        backstory = config['backstory'].format(topic=topic) if '{topic}' in config.get('backstory', '') else config['backstory']
        
        # 🔀 Extract standard agent parameters
        agent_params = {
            'role': role,
            'goal': goal,
            'backstory': backstory,
            'llm': llm_to_use,
            'verbose': config.get('verbose', True),
            'memory': config.get('memory', False),
            'allow_delegation': config.get('allow_delegation', False),
        }
        
        # 🔀 Add voice_style if present
        if 'voice_style' in config:
            agent_params['voice_style'] = config['voice_style']
        
        # 🤖 Create agent with all configured properties
        agents[agent_name.lower()] = Agent(**agent_params)
    
    return agents

# 🔄 Load task configurations and apply topic formatting
def load_tasks(topic, agents, llm_name=None, custom_llm=None):
    """Load task configurations and apply topic formatting"""
    with open("config/tasks.yaml", "r") as file:
        task_configs = yaml.safe_load(file)
    
    # 🤖 Use provided LLM or get a new one
    llm_to_use = custom_llm or get_llm(llm_name)
    
    tasks = []
    for task_name, config in task_configs.items():
        # 🔀 Get the agent specified for this task
        agent_name = config['agent'].lower()  # Convert to lowercase for consistent lookup
        if agent_name not in agents:
            raise ValueError(f"Agent '{agent_name}' specified in task '{task_name}' not found in available agents")
        
        # 🔍 Apply topic formatting to task properties
        description = config['description'].format(topic=topic) if '{topic}' in config.get('description', '') else config['description']
        expected_output = config.get('expected_output', '').format(topic=topic) if '{topic}' in config.get('expected_output', '') else config.get('expected_output', '')
        
        # 🤖 Create task with configured properties and add to tasks list
        tasks.append(Task(
            description=description,      # The instructions for what the agent should accomplish
            expected_output=expected_output,  # Format/structure guidance for the agent's response
            llm=llm_to_use,               # The language model that will process this task
            agent=agents[agent_name]      # Associates this task with the specific agent responsible for it
        ))
    
    return tasks

# 🚀 Main business logic to run a crew with a given topic and LLM
def run_crew_task(topic, llm_name=None):
    """Business logic to run a crew with a given topic and LLM"""
    # 🤖 Get LLM instance
    llm = get_llm(llm_name)
    
    # 🤖 Load agents from configuration with topic
    agents = load_agents(topic, llm_name, llm)
    
    # 🤖 Load tasks from configuration with topic
    tasks = load_tasks(topic, agents, llm_name, llm)
    
    # 🤖 Create the crew with agents and tasks for collaborative AI workflow
    crew = Crew(
        agents=list(agents.values()),  # Converts dictionary of agents to a list for the crew
        tasks=tasks,                   # Assigns the configured tasks to the crew
        process=Process.sequential     # Sets the execution mode to run tasks in order
    )
    
    # 🚀 Execute the crew with the provided topic
    result = crew.kickoff(inputs={"topic": topic})
    return result

# 🧪 Test execution when running script directly
if __name__ == "__main__":
    test_topic = "artificial intelligence"
    result = run_crew_task(test_topic, "gemini_remote")
    print(result)