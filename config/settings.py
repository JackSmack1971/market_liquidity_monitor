"""
Configuration management for market liquidity monitor.

Uses pydantic-settings for type-safe environment variable loading.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenRouter API
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Default model for reasoning
    default_model: str = "anthropic/claude-3.5-sonnet"

    # Exchange API (optional, for authenticated endpoints)
    exchange_api_key: Optional[str] = None
    exchange_api_secret: Optional[str] = None

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # CCXT Configuration
    default_exchange: str = "binance"
    rate_limit: bool = True
    enable_rate_limit: bool = True

    # Agent Configuration
    agent_max_retries: int = 3
    agent_timeout: int = 30

    # Security
    cors_origins: list[str] = ["http://localhost:8501", "http://localhost:3000"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Global settings instance
settings = Settings()
