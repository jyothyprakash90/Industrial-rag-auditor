import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Chunking settings ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Retrieval settings ---
TOP_K = 4  # how many chunks to retrieve per question

# --- Model settings ---
CHAT_MODEL = "gpt-4o-mini"   # cost-efficient, strong quality
EMBEDDING_MODEL = "text-embedding-3-small"
TEMPERATURE = 0

# --- Storage ---
VECTORSTORE_DIR = "vectorstore"
