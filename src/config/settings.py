"""
Module: src.config.settings
Chức năng: Quản lý toàn bộ cấu hình hệ thống từ file .env và môi trường thực thi qua Pydantic Settings.
"""
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Lớp cấu hình tập trung cho toàn bộ ứng dụng On-call & Codebase RAG."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Cấu hình Google Gemini LLM
    gemini_api_key: str = Field(
        default="mock_or_dev_key",
        validation_alias="GEMINI_API_KEY",
        description="API key cho Google Gemini LLM",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        validation_alias="GEMINI_MODEL",
        description="Mã định danh model LLM Gemini",
    )

    # Cấu hình Qdrant Vector Database
    qdrant_host: str = Field(
        default="localhost",
        validation_alias="QDRANT_HOST",
        description="Địa chỉ máy chủ Qdrant",
    )
    qdrant_port: int = Field(
        default=6333,
        validation_alias="QDRANT_PORT",
        description="Cổng HTTP kết nối Qdrant",
    )
    qdrant_grpc_port: int = Field(
        default=6334,
        validation_alias="QDRANT_GRPC_PORT",
        description="Cổng gRPC kết nối Qdrant",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        validation_alias="QDRANT_API_KEY",
        description="API key bảo vệ kết nối Qdrant (nếu có)",
    )

    # Cấu hình Embedding & Reranker Models
    embedding_model_name: str = Field(
        default="BAAI/bge-m3",
        validation_alias="EMBEDDING_MODEL_NAME",
        description="Tên mô hình Hybrid Embedding",
    )
    reranker_model_name: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        validation_alias="RERANKER_MODEL_NAME",
        description="Tên mô hình Cross-Encoder Reranker",
    )

    # Cấu hình GitHub API
    github_token: str | None = Field(
        default=None,
        validation_alias="GITHUB_TOKEN",
        description="Personal access token cho GitHub API",
    )

    # Cấu hình Vận hành & Hệ thống
    log_level: str = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
        description="Mức độ ghi log (DEBUG, INFO, WARNING, ERROR)",
    )
    debug: bool = Field(
        default=False,
        validation_alias="DEBUG",
        description="Chế độ debug (bật trace chi tiết)",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Hàm singleton lấy đối tượng cấu hình Settings duy nhất trong toàn hệ thống."""
    return Settings()
