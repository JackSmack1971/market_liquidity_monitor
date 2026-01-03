"""
Exchange integration using CCXT.

Provides async wrapper around CCXT for non-blocking market data fetching.
"""

import ccxt.async_support as ccxt
from typing import Optional, Dict, Any
import asyncio
from decimal import Decimal

from .models import OrderBook, OrderBookLevel
from ..config import settings


class ExchangeClient:
    """
    Async wrapper for CCXT exchange operations.

    Handles:
    - Async order book fetching
    - Precision handling for exchange-specific requirements
    - Connection lifecycle management
    """

    def __init__(
        self,
        exchange_id: str = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        """
        Initialize exchange client.

        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'coinbase')
            api_key: Optional API key for authenticated requests
            api_secret: Optional API secret
        """
        self.exchange_id = exchange_id or settings.default_exchange
        self.api_key = api_key or settings.exchange_api_key
        self.api_secret = api_secret or settings.exchange_api_secret

        # Initialize exchange instance
        exchange_class = getattr(ccxt, self.exchange_id)
        config = {
            "enableRateLimit": settings.enable_rate_limit,
        }

        if self.api_key and self.api_secret:
            config["apiKey"] = self.api_key
            config["secret"] = self.api_secret

        self.exchange: ccxt.Exchange = exchange_class(config)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources."""
        await self.close()

    async def close(self):
        """Close exchange connection."""
        if self.exchange:
            await self.exchange.close()

    async def fetch_order_book(
        self, symbol: str, limit: int = 20
    ) -> OrderBook:
        """
        Fetch order book for a trading pair.

        Args:
            symbol: Trading pair symbol (e.g., 'SOL/USDT')
            limit: Number of levels to fetch (default 20)

        Returns:
            OrderBook object with structured bid/ask data

        Raises:
            ccxt.NetworkError: If network request fails
            ccxt.ExchangeError: If exchange returns error
        """
        # Fetch raw order book from exchange
        raw_orderbook = await self.exchange.fetch_order_book(symbol, limit)

        # Convert to our structured format
        bids = [
            OrderBookLevel(price=float(bid[0]), amount=float(bid[1]))
            for bid in raw_orderbook["bids"]
        ]

        asks = [
            OrderBookLevel(price=float(ask[0]), amount=float(ask[1]))
            for ask in raw_orderbook["asks"]
        ]

        return OrderBook(
            symbol=symbol,
            exchange=self.exchange_id,
            bids=bids,
            asks=asks,
        )

    async def fetch_markets(self) -> Dict[str, Any]:
        """
        Fetch all available markets on the exchange.

        Returns:
            Dictionary of market information
        """
        return await self.exchange.fetch_markets()

    async def search_symbol(self, query: str) -> list[str]:
        """
        Search for symbols matching a query.

        Args:
            query: Search term (e.g., 'SOL', 'BTC')

        Returns:
            List of matching symbol pairs
        """
        markets = await self.fetch_markets()
        query_upper = query.upper()

        matching_symbols = [
            market["symbol"]
            for market in markets
            if query_upper in market["symbol"]
        ]

        return matching_symbols

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        """
        Format amount to exchange-specific precision.

        Args:
            symbol: Trading pair symbol
            amount: Amount to format

        Returns:
            Formatted amount as string
        """
        return self.exchange.amount_to_precision(symbol, amount)

    def price_to_precision(self, symbol: str, price: float) -> str:
        """
        Format price to exchange-specific precision.

        Args:
            symbol: Trading pair symbol
            price: Price to format

        Returns:
            Formatted price as string
        """
        return self.exchange.price_to_precision(symbol, price)

    async def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed market information for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Market metadata including precision, limits, etc.
        """
        markets = await self.fetch_markets()
        for market in markets:
            if market["symbol"] == symbol:
                return market

        raise ValueError(f"Symbol {symbol} not found on {self.exchange_id}")


class ExchangeManager:
    """
    Manager for multiple exchange clients.

    Provides connection pooling and lifecycle management.
    """

    def __init__(self):
        """Initialize exchange manager."""
        self._clients: Dict[str, ExchangeClient] = {}

    async def get_client(self, exchange_id: str = None) -> ExchangeClient:
        """
        Get or create exchange client.

        Args:
            exchange_id: Exchange identifier

        Returns:
            ExchangeClient instance
        """
        exchange_id = exchange_id or settings.default_exchange

        if exchange_id not in self._clients:
            self._clients[exchange_id] = ExchangeClient(exchange_id)

        return self._clients[exchange_id]

    async def close_all(self):
        """Close all exchange connections."""
        await asyncio.gather(
            *[client.close() for client in self._clients.values()]
        )
        self._clients.clear()


# Global exchange manager instance
exchange_manager = ExchangeManager()
