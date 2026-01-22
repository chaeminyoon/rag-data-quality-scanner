"""
Configuration management using Pydantic Settings.
Loads from environment variables and .env file.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation."""

    # Cohere Configuration
    COHERE_API_KEY: SecretStr = Field(..., description="Cohere API key")
    COHERE_EMBED_MODEL: str = Field(
        default="embed-english-v3.0", description="Cohere embedding model"
    )
    COHERE_RERANK_MODEL: str = Field(
        default="rerank-v3.5", description="Cohere rerank model"
    )

    # Pinecone Configuration
    PINECONE_API_KEY: SecretStr = Field(..., description="Pinecone API key")
    PINECONE_ENVIRONMENT: str = Field(
        default="us-east-1", description="Pinecone environment/region"
    )
    PINECONE_INDEX_NAME: str = Field(
        default="cse-scanner", description="Pinecone index name"
    )

    # Scanner Configuration
    DUPLICATE_THRESHOLD: float = Field(
        default=0.92, ge=0.5, le=1.0, description="Cosine similarity threshold for duplicates"
    )
    MIN_SENTENCE_LENGTH: int = Field(
        default=10, ge=1, description="Minimum sentence length in characters"
    )
    MAX_SENTENCE_LENGTH: int = Field(
        default=2000, ge=100, description="Maximum sentence length in characters"
    )
    BATCH_SIZE: int = Field(
        default=96, ge=1, le=96, description="Batch size for embedding API calls"
    )

    # Evaluation Configuration
    TOP_K: int = Field(default=10, ge=1, le=100, description="Top-K for retrieval evaluation")
    EMBED_DIMENSION: int = Field(default=1024, description="Embedding dimension")

    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()

    @field_validator("EMBED_DIMENSION")
    @classmethod
    def validate_embed_dimension(cls, v: int) -> int:
        valid_dimensions = [256, 384, 512, 768, 1024]
        if v not in valid_dimensions:
            raise ValueError(f"EMBED_DIMENSION must be one of {valid_dimensions}")
        return v


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def validate_api_keys() -> tuple[bool, bool]:
    """
    Validate API keys are present and functional.

    Returns:
        Tuple of (cohere_valid, pinecone_valid)
    """
    settings = get_settings()
    cohere_valid = False
    pinecone_valid = False

    # Test Cohere
    try:
        import cohere

        client = cohere.Client(api_key=settings.COHERE_API_KEY.get_secret_value())
        client.embed(
            texts=["test"],
            model=settings.COHERE_EMBED_MODEL,
            input_type="search_document",
        )
        cohere_valid = True
    except Exception:
        pass

    # Test Pinecone
    try:
        from pinecone import Pinecone

        pc = Pinecone(api_key=settings.PINECONE_API_KEY.get_secret_value())
        pc.list_indexes()
        pinecone_valid = True
    except Exception:
        pass

    return cohere_valid, pinecone_valid
