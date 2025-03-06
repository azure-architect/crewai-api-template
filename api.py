from fastapi import FastAPI, Depends
from pydantic import BaseModel
from main import run_crew_task

app = FastAPI()

# Define the request model with optional llm_name
class TopicRequest(BaseModel):
    topic: str
    llm_name: str = None  # Optional parameter for LLM selection

@app.post("/run-crew/")
async def execute_crew(request: TopicRequest):
    """Endpoint to trigger CrewAI execution with configurable LLM"""
    result = run_crew_task(request.topic, request.llm_name)
    return {"topic": request.topic, "llm": request.llm_name, "result": result}

# Simple health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Run the API: uvicorn api:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)