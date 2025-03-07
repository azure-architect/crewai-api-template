import os
import yaml
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM
from providers import create_llm_from_config
from dotenv import load_dotenv

load_dotenv()

class ResearchRequest(BaseModel):
    topic: str
    llm_name: str = None

def load_llms():
    with open("config/llms.yaml", "r") as file:
        llm_configs = yaml.safe_load(file)
    
    return llm_configs

def get_llm(llm_name=None):
    llm_configs = load_llms()
    
    if not llm_name:
        for name, config in llm_configs.items():
            if config.get("default", False):
                llm_name = name
                break
        if not llm_name and llm_configs:
            llm_name = next(iter(llm_configs))
    
    if llm_name not in llm_configs:
        raise ValueError(f"LLM '{llm_name}' not found in configuration")
    
    config = llm_configs[llm_name]
    
    try:
        return create_llm_from_config(config)
    except ValueError as e:
        raise ValueError(f"Error creating LLM '{llm_name}': {str(e)}")

def load_agents(topic, llm_name=None, custom_llm=None):
    with open("config/agents.yaml", "r") as file:
        agent_configs = yaml.safe_load(file)
    
    llm_to_use = custom_llm or get_llm(llm_name)
    
    agents = {}
    for agent_name, config in agent_configs.items():
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

def load_tasks(topic, agents, llm_name=None, custom_llm=None):
    with open("config/tasks.yaml", "r") as file:
        task_configs = yaml.safe_load(file)
    
    llm_to_use = custom_llm or get_llm(llm_name)
    
    tasks = []
    for task_name, config in task_configs.items():
        agent_name = config['agent']
        if agent_name not in agents:
            raise ValueError(f"Agent '{agent_name}' specified in task '{task_name}' not found in available agents")
        
        description = config['description'].format(topic=topic) if '{topic}' in config.get('description', '') else config['description']
        expected_output = config.get('expected_output', '').format(topic=topic) if '{topic}' in config.get('expected_output', '') else config.get('expected_output', '')
        
        tasks.append(Task(
            description=description,
            expected_output=expected_output,
            agent=agents[agent_name]
        ))
    
    return tasks

def run_crew_task(topic, llm_name=None):
    llm = get_llm(llm_name)
    
    agents = load_agents(topic, llm_name, llm)
    
    tasks = load_tasks(topic, agents, llm_name, llm)
    
    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential
    )
    
    result = crew.kickoff(inputs={"topic": topic})
    return result

if __name__ == "__main__":
    test_topic = "artificial intelligence"
    result = run_crew_task(test_topic, "gemini_remote")
    print(result)