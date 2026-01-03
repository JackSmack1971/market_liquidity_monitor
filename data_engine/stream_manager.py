"""
Stream Manager: Handles high-frequency polling for "Real-Time" data.

Implements the Optimized REST Polling strategy:
1.  Reuse ExchangeClient (for Circuit Breaker protection).
2.  Poll `fetch_order_book` at intervals compliant with Rate Limits.
3.  Write to CacheManager for frontend consumption.
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime
import logfire
from config import settings

from .exchange import exchange_manager
from .cache import cache_manager
from .models import OrderBook, OrderBookLevel

logger = logging.getLogger(__name__)

class StreamManager:
    """
    Manages background polling tasks to simulate real-time streams.
    """

    def __init__(self):
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running = False

    async def start_stream(self, exchange_id: str, symbol: str, depth: int = 20):
        """
        Start a background polling task for the given symbol.
        """
        stream_key = f"{exchange_id}:{symbol}"
        
        if stream_key in self._tasks:
            logger.warning(f"Stream already running for {stream_key}")
            return

        self._running = True
        task = asyncio.create_task(
            self._polling_loop(exchange_id, symbol, depth)
        )
        self._tasks[stream_key] = task
        logger.info(f"Started stream for {stream_key}")

    async def stop_stream(self, exchange_id: str, symbol: str):
        """
        Stop the polling task.
        """
        stream_key = f"{exchange_id}:{symbol}"
        if stream_key in self._tasks:
            self._tasks[stream_key].cancel()
            del self._tasks[stream_key]
            logger.info(f"Stopped stream for {stream_key}")

    async def stop_all(self):
        """Stop all streams."""
        self._running = False
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()

    async def _polling_loop(self, exchange_id: str, symbol: str, depth: int):
        """
        The "Heartbeat" loop: Fetch -> Cache -> Sleep -> Repeat.
        """
        client = await exchange_manager.get_client(exchange_id)
        
        # Determine strict rate limit sleep (in seconds)
        # Default to 1s if undefined, ensure at least 200ms
        rate_limit_ms = getattr(client.exchange, 'rateLimit', 1000)
        sleep_interval = max(rate_limit_ms / 1000.0, 0.2) 

        # Add a small buffer to be safe
        sleep_interval *= 1.1 

        logger.info(f"[{exchange_id}] Polling {symbol} every {sleep_interval:.2f}s")
        
        failures = 0

        while self._running:
            try:
                # 1. Check Circuit Breaker
                if not client.status['is_healthy']:
                    # If circuit is open, wait longer to let it recover/probe
                    await asyncio.sleep(5)
                    continue

                # 2. Fetch Data (Optimized REST)
                # fetch_order_book is already wrapped in circuit breaker inside ExchangeClient
                
                # Logfire Trace
                if settings.logfire_token:
                    with logfire.span("poll_exchange", exchange=exchange_id, symbol=symbol):
                        ob_data = await client.fetch_order_book(symbol, limit=depth)
                else:
                    ob_data = await client.fetch_order_book(symbol, limit=depth)
                
                # 3. Process & Validate
                # Convert raw CCXT response to our Pydantic Model
                # Note: CCXT timestamps are ms
                
                # Simple conversion to dict for caching
                cache_payload = {
                    "exchange": exchange_id,
                    "symbol": symbol,
                    "timestamp": ob_data.get('timestamp') or int(datetime.utcnow().timestamp() * 1000),
                    "datetime": ob_data.get('datetime') or datetime.utcnow().isoformat(),
                    "bids": ob_data.get('bids', [])[:depth],
                    "asks": ob_data.get('asks', [])[:depth],
                    "nonce": ob_data.get('nonce')
                }

                # 4. Push to Hot Cache (Stream Key)
                # TTL = 2 * Sleep Interval (if we miss 2 polls, consider data stale)
                cache_key = f"stream:{exchange_id}:{symbol}"
                # We use a primitive set here because cache_manager might serialize differently. 
                # Let's trust cache_manager.set to handle json serialization if we pass a dict.
                
                await cache_manager.set(
                    key=cache_key,
                    value=cache_payload,
                    ttl=int(sleep_interval * 5) # 5x buffer for safety
                )
                
                failures = 0 # Reset local failure counter

                # 5. Rate Limit Sleep
                await asyncio.sleep(sleep_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                failures += 1
                logger.error(f"Stream Error {exchange_id}/{symbol}: {str(e)}")
                # Backoff strategy
                await asyncio.sleep(min(failures * 1.0, 10.0))

# Global Singleton
stream_manager = StreamManager()
