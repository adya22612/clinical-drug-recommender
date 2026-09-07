"""Configuration for the MCP data server.

Every value has a default so the service starts standalone; docker-compose
overrides the hosts. Required-without-default fields (the previous MONGO_URI
and CHROMA_HOST) crash the import if the env is missing, which turns a config
problem into an unreadable stack trace at container start.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "clinical_db"
    PATIENT_COLLECTION: str = "patients"

    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION: str = "drug_literature"

    # Must match the model used in scripts/ingest_data.py. Differing models
    # put query and document vectors in different spaces; retrieval then
    # returns plausible-looking nonsense with no error raised.
    EMBED_MODEL: str = "all-MiniLM-L6-v2"

    MCP_HOST: str = "0.0.0.0"
    MCP_PORT: int = 8000

    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 20
    MCP_TRANSPORT: str = "sse"   # "stdio" for local testing


settings = Settings()
