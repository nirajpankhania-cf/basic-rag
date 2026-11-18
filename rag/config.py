"""Central settings for the pipeline."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "data" / "policies"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "policies"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking (rationale in rag/chunking.py)
CHUNK_SIZE_WORDS = 150
CHUNK_OVERLAP_WORDS = 30

# Retrieval
TOP_K = 4

# LLM (Azure AI Foundry, OpenAI-compatible v1 endpoint).
# The API key is read from the FOUNDRY_API_KEY environment variable only.
LLM_BASE_URL = os.environ.get(
    "FOUNDRY_BASE_URL", "https://rg-niraj-pankhania.openai.azure.com/openai/v1"
)
LLM_MODEL = os.environ.get("FOUNDRY_MODEL", "gpt-6.1-sol")
