"""
Redis-based caching layer for market data.

Provides:
- Order book caching with TTL
- Historical data caching
- Query result caching
- Cache invalidation strategies
"""

import json
import asyncio
from typing import Optional, Any, Callable
from datetime import datetime, timedelta
from functools import wraps
import hashlib

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: redis package not available. Caching disabled.")


class CacheManager:
    """
    Async Redis cache manager for market data.

    Falls back to in-memory cache if Redis unavailable.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int = 60,
        use_redis: bool = True
    ):
        """
        Initialize cache manager.

        Args:
            redis_url: Redis connection URL
            default_ttl: Default cache TTL in seconds
            use_redis: Whether to use Redis (False = in-memory only)
        """
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.use_redis = use_redis and REDIS_AVAILABLE
        self._redis: Optional[redis.Redis] = None
        self._memory_cache: dict = {}  # Fallback in-memory cache
        self._memory_cache_expiry: dict = {}

    async def connect(self):
        """Connect to Redis."""
        if self.use_redis and not self._redis:
            try:
                self._redis = await redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                # Test connection
                await self._redis.ping()
                print(f"✅ Connected to Redis at {self.redis_url}")
            except Exception as e:
                print(f"⚠️ Redis connection failed: {e}. Using in-memory cache.")
                self._redis = None
                self.use_redis = False

    async def disconnect(self):
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    def _make_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate cache key from arguments.

        Args:
            prefix: Key prefix (e.g., 'orderbook', 'comparison')
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        # Create deterministic key from arguments
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:12]
        return f"mlm:{prefix}:{key_hash}"

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        # Try Redis first
        if self._redis:
            try:
                value = await self._redis.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                print(f"Redis get error: {e}")

        # Fallback to memory cache
        if key in self._memory_cache:
            # Check expiry
            if key in self._memory_cache_expiry:
                if datetime.utcnow() < self._memory_cache_expiry[key]:
                    return self._memory_cache[key]
                else:
                    # Expired, remove
                    del self._memory_cache[key]
                    del self._memory_cache_expiry[key]

        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time-to-live in seconds (None = use default)
        """
        ttl = ttl or self.default_ttl

        # Try Redis first
        if self._redis:
            try:
                await self._redis.setex(
                    key,
                    ttl,
                    json.dumps(value, default=str)
                )
                return
            except Exception as e:
                print(f"Redis set error: {e}")

        # Fallback to memory cache
        self._memory_cache[key] = value
        self._memory_cache_expiry[key] = datetime.utcnow() + timedelta(seconds=ttl)

    async def delete(self, key: str):
        """
        Delete key from cache.

        Args:
            key: Cache key
        """
        # Redis
        if self._redis:
            try:
                await self._redis.delete(key)
            except Exception as e:
                print(f"Redis delete error: {e}")

        # Memory
        if key in self._memory_cache:
            del self._memory_cache[key]
        if key in self._memory_cache_expiry:
            del self._memory_cache_expiry[key]

    async def clear_pattern(self, pattern: str):
        """
        Clear all keys matching pattern.

        Args:
            pattern: Key pattern (e.g., 'mlm:orderbook:*')
        """
        if self._redis:
            try:
                keys = await self._redis.keys(pattern)
                if keys:
                    await self._redis.delete(*keys)
            except Exception as e:
                print(f"Redis clear_pattern error: {e}")

        # Memory cache - simple prefix match
        keys_to_delete = [
            k for k in self._memory_cache.keys()
            if pattern.replace('*', '') in k
        ]
        for key in keys_to_delete:
            if key in self._memory_cache:
                del self._memory_cache[key]
            if key in self._memory_cache_expiry:
                del self._memory_cache_expiry[key]

    async def cache_orderbook(
        self,
        symbol: str,
        exchange: str,
        orderbook,
        ttl: int = 5
    ):
        """
        Cache order book data.

        Args:
            symbol: Trading pair
            exchange: Exchange name
            orderbook: OrderBook object
            ttl: Cache TTL in seconds (default 5s for real-time data)
        """
        key = self._make_key("orderbook", symbol=symbol, exchange=exchange)
        await self.set(key, orderbook.model_dump(mode='json'), ttl=ttl)

    async def get_orderbook(
        self,
        symbol: str,
        exchange: str
    ) -> Optional[dict]:
        """
        Get cached order book.

        Args:
            symbol: Trading pair
            exchange: Exchange name

        Returns:
            OrderBook dict or None
        """
        key = self._make_key("orderbook", symbol=symbol, exchange=exchange)
        return await self.get(key)

    async def cache_comparison(
        self,
        symbol: str,
        exchanges: list,
        comparison_data: dict,
        ttl: int = 10
    ):
        """
        Cache multi-exchange comparison.

        Args:
            symbol: Trading pair
            exchanges: List of exchanges
            comparison_data: Comparison result
            ttl: Cache TTL in seconds
        """
        key = self._make_key(
            "comparison",
            symbol=symbol,
            exchanges=tuple(sorted(exchanges))
        )
        await self.set(key, comparison_data, ttl=ttl)

    async def get_comparison(
        self,
        symbol: str,
        exchanges: list
    ) -> Optional[dict]:
        """Get cached comparison data."""
        key = self._make_key(
            "comparison",
            symbol=symbol,
            exchanges=tuple(sorted(exchanges))
        )
        return await self.get(key)

    async def invalidate_symbol(self, symbol: str):
        """
        Invalidate all cache entries for a symbol.

        Args:
            symbol: Trading pair
        """
        await self.clear_pattern(f"mlm:*:{symbol}:*")

    async def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        stats = {
            "backend": "redis" if self._redis else "memory",
            "memory_keys": len(self._memory_cache),
        }

        if self._redis:
            try:
                info = await self._redis.info("stats")
                stats.update({
                    "redis_connected": True,
                    "redis_keys": await self._redis.dbsize(),
                    "redis_hits": info.get("keyspace_hits", 0),
                    "redis_misses": info.get("keyspace_misses", 0),
                })
            except Exception as e:
                stats["redis_error"] = str(e)

        return stats


# Decorator for caching function results
def cached(
    cache_manager: CacheManager,
    prefix: str,
    ttl: int = 60,
    key_args: Optional[list] = None
):
    """
    Decorator to cache async function results.

    Args:
        cache_manager: CacheManager instance
        prefix: Cache key prefix
        ttl: Time-to-live in seconds
        key_args: List of argument names to include in cache key

    Example:
        @cached(cache_manager, prefix="market_data", ttl=30, key_args=["symbol"])
        async def fetch_market_data(symbol: str, exchange: str):
            # Expensive operation
            return data
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key from specified arguments
            if key_args:
                key_kwargs = {k: kwargs.get(k) for k in key_args if k in kwargs}
            else:
                key_kwargs = kwargs

            cache_key = cache_manager._make_key(prefix, *args, **key_kwargs)

            # Try to get from cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            await cache_manager.set(cache_key, result, ttl=ttl)

            return result

        return wrapper
    return decorator


# Global cache manager instance
cache_manager = CacheManager(
    redis_url="redis://localhost:6379",
    default_ttl=60,
    use_redis=True  # Will fall back to memory if Redis unavailable
)


# Example usage with historical data
async def cache_snapshot(symbol: str, exchange: str, snapshot):
    """Cache historical snapshot."""
    key = f"mlm:snapshot:{exchange}:{symbol.replace('/', '_')}:{snapshot.timestamp.isoformat()}"
    await cache_manager.set(key, snapshot.model_dump(mode='json'), ttl=86400)  # 24h TTL


async def get_recent_snapshots(
    symbol: str,
    exchange: str,
    hours: int = 24
) -> list:
    """
    Get recent snapshots from cache.

    Args:
        symbol: Trading pair
        exchange: Exchange name
        hours: Lookback period

    Returns:
        List of cached snapshots
    """
    # This is a simplified version - in production, use Redis sorted sets
    # for efficient time-range queries
    pattern = f"mlm:snapshot:{exchange}:{symbol.replace('/', '_')}:*"

    if cache_manager._redis:
        keys = await cache_manager._redis.keys(pattern)
        snapshots = []
        for key in keys:
            data = await cache_manager.get(key)
            if data:
                snapshots.append(data)
        return snapshots
    else:
        return []
