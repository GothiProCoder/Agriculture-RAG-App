import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Central configuration for the RAG application."""
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Model Configurations
    LLM_MODEL = "gemini-1.5-flash"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Retrieval Configurations
    BM25_K = 30
    FAISS_K = 30
    RERANK_THRESHOLD = -4.0