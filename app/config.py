from functools import lru_cache
import json

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Cents"
    database_url: str = "sqlite+aiosqlite:///./cents.db"
    jwt_secret: str = "change-me"
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:4200", "http://127.0.0.1:4200"]
    )
    vector_dimension: int = 1536
    chroma_persist_directory: str = ".chroma"
    chroma_documents_collection: str = "documents"
    chroma_tools_collection: str = "tool_definitions"
    llm_service_base_url: str = "http://127.0.0.1:8100"
    llm_service_generate_path: str = "/v1/generate"
    llm_service_models_path: str = "/v1/models"
    llm_service_embeddings_path: str = "/v1/embeddings"
    llm_service_api_key: str = ""
    llm_service_timeout_seconds: int = Field(default=120, ge=5, le=600)
    llm_default_generation_model_type: str = "text-generation"
    llm_default_generation_model: str = ""
    llm_default_embedding_model_type: str = "embedding"
    llm_default_embedding_model: str = ""
    llm_default_judge_model_type: str = "reasoning"
    llm_default_judge_model: str = ""
    llm_judge_enabled: bool = False
    llm_default_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    llm_default_max_tokens: int = Field(default=512, ge=1, le=4096)
    service_call_allowed_hosts: list[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1"])
    service_call_allow_unsafe_destinations: bool = False
    service_call_default_timeout_seconds: int = Field(default=30, ge=1, le=600)
    service_call_secrets: dict[str, str] = Field(default_factory=dict)
    tool_code_execution_timeout_seconds: int = Field(default=20, ge=1, le=120)
    tool_code_max_length: int = Field(default=120000, ge=1000, le=1000000)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return [str(origin).strip() for origin in parsed if str(origin).strip()]
                except json.JSONDecodeError:
                    pass
            return [origin.strip() for origin in stripped.split(",") if origin.strip()]
        return value

    @field_validator("service_call_allowed_hosts", mode="before")
    @classmethod
    def parse_service_call_allowed_hosts(cls, value):
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return [str(host).strip().lower() for host in parsed if str(host).strip()]
                except json.JSONDecodeError:
                    pass
            return [host.strip().lower() for host in stripped.split(",") if host.strip()]
        if isinstance(value, list):
            return [str(host).strip().lower() for host in value if str(host).strip()]
        return value

    @field_validator("service_call_secrets", mode="before")
    @classmethod
    def parse_service_call_secrets(cls, value):
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return {}
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    return {
                        str(key).strip(): str(secret)
                        for key, secret in parsed.items()
                        if str(key).strip()
                    }
            except json.JSONDecodeError:
                return {}
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
