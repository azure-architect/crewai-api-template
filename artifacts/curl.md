# Check if Ollama is running
curl -s http://localhost:11434/api/version


curl -X POST http://localhost:8000/run-crew/ \
  -H "Content-Type: application/json" \
  -d '{"topic": "quantum computing", "llm_name": "gemini_remote"}'


curl -X POST http://localhost:8000/run-crew/ \
  -H "Content-Type: application/json" \
  -d '{"topic": "renewable energy", "llm_name": "llama_local"}'