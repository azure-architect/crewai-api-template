from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3:8b", base_url="http://localhost:11434")

response = llm.invoke("Tell me a joke.")
print(response)
