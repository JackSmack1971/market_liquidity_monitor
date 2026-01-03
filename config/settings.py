"""
Configuration management for market liquidity monitor.

Uses pydantic-settings for type-safe environment variable loading.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenRouter API
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Database URL
    database_url: str = "postgresql+asyncpg://mlm_user:mlm_password@localhost:5432/market_liquidity"

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
    
    # Observability (Logfire)
    logfire_token: Optional[str] = None
    logfire_service_name: str = "market-liquidity-monitor"
    logfire_environment: str = "development"
    logfire_trace_sample_rate: float = 0.2  # 0.2 = 20% of traces for quota protection
    logfire_console_level: str = "info"  # 'info', 'warn', 'error'
    logfire_auto_trace_min_duration: float = 0.075  # 75ms to ignore trivial traces

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Global settings instance
settings = Settings()
