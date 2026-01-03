"""
Dependency injection for FastAPI.

Provides reusable dependencies for exchange clients and agents.
"""

from typing import AsyncGenerator
from fastapi import Depends

from data_engine import ExchangeClient, exchange_manager
from agents import market_analyzer, MarketAnalyzer


async def get_exchange_client(
    exchange_id: str = "binance",
) -> AsyncGenerator[ExchangeClient, None]:
    """
    Dependency to get an exchange client.

    Args:
        exchange_id: Exchange identifier

    Yields:
        ExchangeClient instance
    """
    client = await exchange_manager.get_client(exchange_id)
    yield client


async def get_market_analyzer() -> MarketAnalyzer:
    """
    Dependency to get the market analyzer.

    Returns:
        MarketAnalyzer instance
    """
    return market_analyzer
