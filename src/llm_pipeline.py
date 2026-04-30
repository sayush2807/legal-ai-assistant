import requests
import json

from src.embed_store import query_vector_store

def generate_answer(query, top_k=3, model="llama3"):
    # First get retrievals
    retrieved = query_vector_store(query, top_k)
    context = "\n\n".join(retrieved["documents"][0])

    prompt = f"""
    You are a legal research assistant.
    Answer the following question using ONLY the provided context.
    Cite sources at the end.
    
    Question: {query}

    Context:
    {context}
    """

    # Call Ollama API (local server)
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt}

    response = requests.post(url, json=payload, stream=True)

    answer = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                answer += data["response"]

    return answer.strip()
