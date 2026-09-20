"""Environment configuration. Reads everything from .env via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"

    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 500
    llm_temperature: float = 0.2

    chroma_path: str = "./data/chroma"
    collection_name: str = "company_docs"
    top_k: int = 5
    data_dir: str = "./data/docs"

    chunk_size: int = 500
    chunk_overlap: int = 100


settings = Settings()