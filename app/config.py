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
    llm_service_api_key: str = ""
    llm_service_timeout_seconds: int = Field(default=120, ge=5, le=600)
    llm_default_generation_model_type: str = "text-generation"
    llm_default_generation_model: str = ""
    llm_default_judge_model_type: str = "reasoning"
    llm_default_judge_model: str = ""
    llm_judge_enabled: bool = False
    llm_default_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    llm_default_max_tokens: int = Field(default=512, ge=1, le=4096)

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


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
