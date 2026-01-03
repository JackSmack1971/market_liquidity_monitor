import ccxt.async_support as ccxt
from typing import Optional, Dict, Any, Callable, List
import asyncio
import time
from decimal import Decimal
from functools import wraps
from data_engine.models import OrderBook, OrderBookLevel
from data_engine.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from config import settings


def with_retry(retries: int = 3, backoff: float = 1.0):
    """
    Decorator for robust CCXT request retries.
    Handles NetworkError, RateLimitExceeded, and DDoSProtection.
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            last_error = None
            for attempt in range(retries):
                try:
                    return await func(self, *args, **kwargs)
                except (ccxt.RateLimitExceeded, ccxt.DDoSProtection) as e:
                    last_error = e
                    # Mandatory 10s sleep for rate limits/DDoS protection
                    delay = 10.0
                    print(f"⚠️ Rate limit hit on {self.exchange_id}, sleeping {delay}s (Attempt {attempt + 1}/{retries})")
                    await asyncio.sleep(delay)
                except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                    last_error = e
                    delay = backoff * (2 ** attempt)
                    print(f"Network error on {self.exchange_id}, retrying in {delay}s... ({e})")
                    await asyncio.sleep(delay)
                except Exception as e:
                    # Don't retry on other errors
                    raise e
            raise last_error
        return wrapper
    return decorator


class ExchangeClient:
    """
    Async wrapper for CCXT exchange operations.

    Handles:
    - Async order book fetching with retry logic
    - Precision handling for exchange-specific requirements
    - Market metadata loading and limit validation
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
        self._markets_loaded = False

        # Initialize exchange instance
        exchange_class = getattr(ccxt, self.exchange_id)
        config = {
            "enableRateLimit": True,  # Mandatory for production stability
            "adjustForTimeDifference": True,  # Prevent timestamp ahead of server errors
        }

        if self.api_key and self.api_secret:
            config["apiKey"] = self.api_key
            config["secret"] = self.api_secret

        self.exchange: ccxt.Exchange = exchange_class(config)
        
        # Circuit Breaker for this exchange
        self.circuit_breaker = CircuitBreaker(
            name=self.exchange_id,
            failure_threshold=5,
            recovery_timeout=30
        )

    @property
    def status(self) -> dict:
        """Get exchange connection health status."""
        return {
            "name": self.exchange_id,
            "state": self.circuit_breaker.state,
            "failures": self.circuit_breaker._failures,
            "is_healthy": self.circuit_breaker.state == "CLOSED"
        }

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
            print(f"🔌 Connection closed for {self.exchange_id}")

    async def ensure_markets_loaded(self):
        """Ensure market metadata is loaded before operations."""
        if not self._markets_loaded:
            # Avoid circular import by importing here if needed, 
            # but we already use absolute imports in the manager.
            from data_engine.exchange import exchange_manager
            if self.exchange_id in exchange_manager._markets_cache:
                self.exchange.markets = exchange_manager._markets_cache[self.exchange_id]
                self._markets_loaded = True
            else:
                print(f"Loading markets for {self.exchange_id}...")
                await self.exchange.load_markets()
                exchange_manager._markets_cache[self.exchange_id] = self.exchange.markets
                self._markets_loaded = True

    @with_retry()
    async def fetch_order_book(
        self, symbol: str, limit: int = 20
    ) -> OrderBook:
        """
        Fetch order book for a trading pair with retry logic.

        Args:
            symbol: Trading pair symbol (e.g., 'SOL/USDT')
            limit: Number of levels to fetch (default 20)

        Returns:
            OrderBook object with structured bid/ask data
        """
        await self.ensure_markets_loaded()
        
        # Fetch with Circuit Breaker
        raw_orderbook = await self.circuit_breaker.call(
            self.exchange.fetch_order_book, 
            symbol, 
            limit
        )

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

    @with_retry()
    async def fetch_markets(self) -> Dict[str, Any]:
        """
        Fetch all available markets on the exchange.

        Returns:
            Dictionary of market information
        """
        return await self.exchange.fetch_markets()

    @with_retry()
    async def fetch_ohlcv(
        self, 
        symbol: str, 
        timeframe: str = '1h', 
        since: Optional[int] = None, 
        limit: Optional[int] = None
    ) -> List[list]:
        """
        Fetch OHLCV (candlestick) data.

        Args:
            symbol: Trading pair symbol
            timeframe: Candle duration ('1m', '5m', '1h', '1d')
            since: Timestamp in ms
            limit: Number of candles

        Returns:
            List of [timestamp, open, high, low, close, volume]
        """
        await self.ensure_markets_loaded()
        
        if not self.exchange.has['fetchOHLCV']:
            raise NotImplementedError(f"Exchange {self.exchange_id} does not support OHLCV fetching.")
            
        # Try Cache
        from data_engine.cache import cache_manager
        cache_key = f"ohlcv:{self.exchange_id}:{symbol}:{timeframe}:{since}:{limit}"
        cached_data = await cache_manager.get(cache_key)
        if cached_data:
            return cached_data

        # Fetch with Circuit Breaker
        data = await self.circuit_breaker.call(
            self.exchange.fetch_ohlcv, 
            symbol, 
            timeframe, 
            since, 
            limit
        )
        
        # Set Cache (5 min TTL)
        await cache_manager.set(cache_key, data, ttl=300)
        
        return data

    async def search_symbol(self, query: str) -> list[str]:
        """
        Search for symbols matching a query.

        Args:
            query: Search term (e.g., 'SOL', 'BTC')

        Returns:
            List of matching symbol pairs
        """
        await self.ensure_markets_loaded()
        markets = self.exchange.markets
        query_upper = query.upper()

        matching_symbols = [
            market["symbol"]
            for market in markets.values()
            if query_upper in market["symbol"]
        ]

        return matching_symbols

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        """
        Format amount to exchange-specific precision.
        """
        return self.exchange.amount_to_precision(symbol, amount)

    def price_to_precision(self, symbol: str, price: float) -> str:
        """
        Format price to exchange-specific precision.
        """
        return self.exchange.price_to_precision(symbol, price)

    def cost_to_precision(self, symbol: str, cost: float) -> str:
        """
        Format cost to exchange-specific precision.
        """
        return self.exchange.cost_to_precision(symbol, cost)

    def validate_order_limits(self, symbol: str, amount: float, price: float) -> tuple[bool, str]:
        """
        Validate that order amount and cost satisfy exchange limits.

        Args:
            symbol: Trading pair symbol
            amount: Order amount
            price: Order price

        Returns:
            (is_valid, error_message)
        """
        if not self._markets_loaded:
            return True, ""
            
        try:
            market = self.exchange.market(symbol)
        except Exception:
            return False, f"Symbol {symbol} not found on {self.exchange_id}"

        limits = market.get('limits', {})
        
        # Check amount limits
        amount_limits = limits.get('amount', {})
        if amount_limits.get('min') is not None and amount < amount_limits['min']:
            return False, f"Order amount {amount} is below the minimum required ({amount_limits['min']}) for {symbol}"
        if amount_limits.get('max') is not None and amount > amount_limits['max']:
            return False, f"Order amount {amount} exceeds the maximum allowed ({amount_limits['max']}) for {symbol}"
            
        # Check cost limits (amount * price)
        cost = amount * price
        cost_limits = limits.get('cost', {})
        if cost_limits.get('min') is not None and cost < cost_limits['min']:
            return False, f"Order cost ${cost:,.2f} is below the minimum required (${cost_limits['min']:,.2f}) for {symbol}"
            
        return True, ""

    async def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed market information for a symbol.
        """
        await self.ensure_markets_loaded()
        if symbol in self.exchange.markets:
            return self.exchange.markets[symbol]

        raise ValueError(f"Symbol {symbol} not found on {self.exchange_id}")


class ExchangeManager:
    """
    Manager for multiple exchange clients.

    Provides connection pooling and lifecycle management.
    """

    def __init__(self):
        """Initialize exchange manager."""
        self._clients: Dict[str, ExchangeClient] = {}
        self._markets_cache: Dict[str, Dict[str, Any]] = {}

    async def preload_exchange(self, exchange_id: str):
        """Pre-load markets for an exchange to warm up the cache."""
        async with ExchangeClient(exchange_id) as client:
            await client.ensure_markets_loaded()
            self._markets_cache[exchange_id] = client.exchange.markets
            print(f"✅ Market cache warmed for {exchange_id}")

    async def get_client(self, exchange_id: str = None) -> ExchangeClient:
        """
        Get or create pooled exchange client.
        Ensures stable rate limit buckets across requests.
        """
        exchange_id = exchange_id or settings.default_exchange
        if exchange_id not in self._clients:
            print(f"🚀 Initializing pooled client for {exchange_id}")
            self._clients[exchange_id] = ExchangeClient(exchange_id)
        return self._clients[exchange_id]

    async def close_all(self):
        """Close all pooled exchange connections."""
        for exchange_id, client in self._clients.items():
            await client.close()
        self._markets_cache.clear()
        self._clients.clear()
        print("✅ All exchange connections closed.")


# Global exchange manager instance
exchange_manager = ExchangeManager()
