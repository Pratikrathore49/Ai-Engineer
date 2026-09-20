"""Central configuration from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 600

    chroma_path: str = "./data/chroma"
    collection_name: str = "company_docs"
    data_dir: str = "./data/docs"

    top_k: int = 5
    retrieve_candidates: int = 20
    bm25_weight: float = 0.5
    min_relevance_score: float = 0.35
    rerank_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_size: int = 500
    chunk_overlap: int = 100

    api_key: str = "change-me-in-prod"
    cache_ttl_seconds: int = 300
    rate_limit_per_min: int = 60
    max_upload_mb: int = 10
    log_level: str = "INFO"


settings = Settings()