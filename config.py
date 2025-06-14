import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "service_name": os.getenv("DB_SERVICE_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "schema": os.getenv("DB_SCHEMA"),
}

# API keys
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# File paths
SCHEMA_FILE = "table_details.json"
VECTOR_INDEX_FILE = "faiss_index.bin"
VECTOR_NAMES_FILE = "table_names.json"
VECTOR_DESC_FILE = "table_descriptions.json"
LAST_FETCH_FILE = "last_fetch.txt"

# Available models
MISTRAL_MODELS = ["mistral-large-latest", "mistral-medium", "mistral-small"]

# Embedding model
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"