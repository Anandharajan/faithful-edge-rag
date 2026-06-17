from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "faithful-edge-rag"
    app_env: str = "local"
    log_level: str = "INFO"
    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379/0"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
