#!/usr/bin/env python3
"""
Test script for CrewAI with CustomGeminiLLM
"""

import os
from crewai import Agent, Task, Crew
from gemini_llm import CustomGeminiLLM

# Ensure API key is set
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY environment variable is not set.")
    print("Run: export GEMINI_API_KEY=your_api_key")
    exit(1)

# Initialize the Gemini LLM
gemini_llm = CustomGeminiLLM(
    api_key=api_key,
    temperature=0.7,
    max_tokens=2048
)

# Create a simple agent
researcher = Agent(
    role="Researcher",
    goal="Research and summarize information",
    backstory="You are an expert researcher who can find and summarize information effectively.",
    verbose=True,
    llm=gemini_llm
)

# Create a simple task
research_task = Task(
    description="Summarize the key features of large language models in 3-5 bullet points.",
    agent=researcher
)

# Create the crew with the agent and task
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=2
)

# Run the crew and get the result
result = crew.kickoff()

print("\n=== CREW RESULT ===")
print(result)