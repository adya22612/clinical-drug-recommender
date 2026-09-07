"""Configuration for the evaluation harness."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Evaluation runs from the host, so these are published container ports
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "clinical_db"
    PATIENT_COLLECTION: str = "patients"

    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION: str = "drug_literature"
    EMBED_MODEL: str = "all-MiniLM-L6-v2"

    ORCHESTRATOR_URL: str = "http://localhost:8080"

    # A four-tool ReAct loop plus synthesis on CPU takes minutes per case
    REQUEST_TIMEOUT: float = 900.0

    # Judge model for RAGAS. A 3B judge is weak - see README.
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    JUDGE_MODEL: str = "llama3.2:3b"


settings = Settings()
