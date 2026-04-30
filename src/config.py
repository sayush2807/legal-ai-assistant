import os

# Set your API keys here (or use environment variables)
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # fallback if no OpenAI
VECTOR_DB_PATH = "data/vector_store"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200