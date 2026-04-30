from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from src.config import EMBEDDING_MODEL,VECTOR_DB_PATH

def create_vector_store(docs):
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    collection = client.get_or_create_collection("legal_docs")

    for i, doc in enumerate(docs):
        emb = model.encode(doc).tolist()
        collection.add(documents=[doc], embeddings=[emb], ids=[str(i)])

    return collection

def query_vector_store(query, top_k=3):
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = client.get_collection("legal_docs")

    q_emb = model.encode(query).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    return results