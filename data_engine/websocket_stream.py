"""
WebSocket streaming for real-time order book updates.

Provides low-latency streaming of order book changes using exchange WebSocket APIs.
"""

import asyncio
import json
from typing import Optional, Callable, Dict, Set
from datetime import datetime
import ccxt.async_support as ccxt

from .models import OrderBook, OrderBookLevel


class OrderBookStream:
    """
    Real-time order book streaming via WebSocket.

    Maintains live order book state and notifies subscribers of updates.
    """

    def __init__(self, exchange_id: str = "binance"):
        """
        Initialize order book stream.

        Args:
            exchange_id: Exchange identifier
        """
        self.exchange_id = exchange_id
        self.exchange: Optional[ccxt.Exchange] = None
        self._subscribers: Dict[str, Set[Callable]] = {}
        self._running = False
        self._tasks: Dict[str, asyncio.Task] = {}

    async def _initialize_exchange(self):
        """Initialize exchange with WebSocket support."""
        if not self.exchange:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                "enableRateLimit": True,
                "newUpdates": True,  # Enable WebSocket mode
            })

    async def subscribe(
        self,
        symbol: str,
        callback: Callable[[OrderBook], None],
        depth: int = 20
    ):
        """
        Subscribe to order book updates for a symbol.

        Args:
            symbol: Trading pair (e.g., 'SOL/USDT')
            callback: Function to call on each update
            depth: Order book depth to maintain

        Example:
            async def handle_update(orderbook: OrderBook):
                print(f"Spread: {orderbook.spread_percentage}%")

            stream = OrderBookStream("binance")
            await stream.subscribe("SOL/USDT", handle_update)
            await stream.start()
        """
        await self._initialize_exchange()

        # Add subscriber
        if symbol not in self._subscribers:
            self._subscribers[symbol] = set()
        self._subscribers[symbol].add(callback)

        # Start streaming task if not already running
        if symbol not in self._tasks:
            task = asyncio.create_task(
                self._stream_orderbook(symbol, depth)
            )
            self._tasks[symbol] = task

    def unsubscribe(self, symbol: str, callback: Callable):
        """
        Unsubscribe from order book updates.

        Args:
            symbol: Trading pair
            callback: Callback function to remove
        """
        if symbol in self._subscribers:
            self._subscribers[symbol].discard(callback)

            # Cancel task if no more subscribers
            if not self._subscribers[symbol] and symbol in self._tasks:
                self._tasks[symbol].cancel()
                del self._tasks[symbol]

    async def _stream_orderbook(self, symbol: str, depth: int):
        """
        Internal method to stream order book updates.

        Args:
            symbol: Trading pair
            depth: Order book depth
        """
        while True:
            try:
                # Check if exchange supports watch_order_book
                if hasattr(self.exchange, 'watch_order_book'):
                    # Use native WebSocket streaming
                    orderbook_data = await self.exchange.watch_order_book(
                        symbol,
                        limit=depth
                    )
                else:
                    # Fallback to polling if WebSocket not supported
                    orderbook_data = await self.exchange.fetch_order_book(
                        symbol,
                        limit=depth
                    )
                    await asyncio.sleep(1)  # Throttle polling

                # Convert to our OrderBook model
                bids = [
                    OrderBookLevel(price=float(bid[0]), amount=float(bid[1]))
                    for bid in orderbook_data.get("bids", [])
                ]
                asks = [
                    OrderBookLevel(price=float(ask[0]), amount=float(ask[1]))
                    for ask in orderbook_data.get("asks", [])
                ]

                orderbook = OrderBook(
                    symbol=symbol,
                    exchange=self.exchange_id,
                    timestamp=datetime.utcnow(),
                    bids=bids,
                    asks=asks,
                )

                # Notify all subscribers
                if symbol in self._subscribers:
                    for callback in self._subscribers[symbol]:
                        try:
                            # Handle both sync and async callbacks
                            if asyncio.iscoroutinefunction(callback):
                                await callback(orderbook)
                            else:
                                callback(orderbook)
                        except Exception as e:
                            print(f"Error in callback for {symbol}: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error streaming {symbol}: {e}")
                await asyncio.sleep(5)  # Wait before retry

    async def start(self):
        """Start the WebSocket stream."""
        self._running = True
        await self._initialize_exchange()

    async def stop(self):
        """Stop the WebSocket stream and cleanup."""
        self._running = False

        # Cancel all streaming tasks
        for task in self._tasks.values():
            task.cancel()

        # Wait for tasks to finish
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)

        # Close exchange connection
        if self.exchange:
            await self.exchange.close()

        self._tasks.clear()
        self._subscribers.clear()


class MultiExchangeStream:
    """
    Aggregate WebSocket streams from multiple exchanges.

    Useful for real-time arbitrage detection and cross-exchange monitoring.
    """

    def __init__(self, exchange_ids: list[str]):
        """
        Initialize multi-exchange stream.

        Args:
            exchange_ids: List of exchange identifiers
        """
        self.streams = {
            exchange_id: OrderBookStream(exchange_id)
            for exchange_id in exchange_ids
        }
        self._aggregated_data: Dict[str, Dict[str, OrderBook]] = {}

    async def subscribe(
        self,
        symbol: str,
        callback: Callable[[Dict[str, OrderBook]], None]
    ):
        """
        Subscribe to order book updates across all exchanges.

        Args:
            symbol: Trading pair
            callback: Function called with dict of {exchange: orderbook}

        Example:
            async def handle_multi(data: Dict[str, OrderBook]):
                binance = data.get("binance")
                coinbase = data.get("coinbase")
                if binance and coinbase:
                    arb = coinbase.best_bid.price - binance.best_ask.price
                    if arb > 0:
                        print(f"Arbitrage opportunity: ${arb:.2f}")

            stream = MultiExchangeStream(["binance", "coinbase"])
            await stream.subscribe("SOL/USDT", handle_multi)
            await stream.start()
        """
        # Initialize aggregated data structure
        if symbol not in self._aggregated_data:
            self._aggregated_data[symbol] = {}

        # Subscribe to each exchange
        for exchange_id, stream in self.streams.items():
            async def exchange_callback(orderbook: OrderBook, ex_id=exchange_id):
                # Update aggregated data
                self._aggregated_data[symbol][ex_id] = orderbook

                # Call user callback with all data
                await callback(self._aggregated_data[symbol])

            await stream.subscribe(symbol, exchange_callback)

    async def start(self):
        """Start all exchange streams."""
        await asyncio.gather(*[
            stream.start()
            for stream in self.streams.values()
        ])

    async def stop(self):
        """Stop all exchange streams."""
        await asyncio.gather(*[
            stream.stop()
            for stream in self.streams.values()
        ])


# Example: Liquidity Monitor with Alerts
class LiveLiquidityMonitor:
    """
    Real-time liquidity monitoring with anomaly detection.

    Combines WebSocket streaming with alert generation.
    """

    def __init__(
        self,
        symbol: str,
        exchange: str = "binance",
        spread_threshold_pct: float = 0.5,
        depth_threshold_pct: float = 30.0
    ):
        """
        Initialize live monitor.

        Args:
            symbol: Trading pair to monitor
            exchange: Exchange name
            spread_threshold_pct: Alert if spread exceeds this percentage
            depth_threshold_pct: Alert if depth drops by this percentage
        """
        self.symbol = symbol
        self.exchange = exchange
        self.spread_threshold = spread_threshold_pct
        self.depth_threshold = depth_threshold_pct

        self.stream = OrderBookStream(exchange)
        self.baseline_depth: Optional[float] = None
        self.alert_callbacks: list[Callable] = []

    def on_alert(self, callback: Callable[[dict], None]):
        """
        Register callback for alerts.

        Args:
            callback: Function to call when alert triggers
        """
        self.alert_callbacks.append(callback)

    async def _handle_update(self, orderbook: OrderBook):
        """Handle order book update."""
        # Initialize baseline
        if self.baseline_depth is None:
            self.baseline_depth = orderbook.get_cumulative_volume("bids", 10)

        # Check spread
        if orderbook.spread_percentage and orderbook.spread_percentage > self.spread_threshold:
            alert = {
                "type": "SPREAD_ALERT",
                "symbol": self.symbol,
                "exchange": self.exchange,
                "spread_pct": orderbook.spread_percentage,
                "threshold": self.spread_threshold,
                "timestamp": orderbook.timestamp.isoformat(),
                "message": f"Spread widened to {orderbook.spread_percentage:.3f}%"
            }
            await self._trigger_alerts(alert)

        # Check depth
        current_depth = orderbook.get_cumulative_volume("bids", 10)
        if self.baseline_depth:
            depth_drop = ((self.baseline_depth - current_depth) / self.baseline_depth) * 100
            if depth_drop > self.depth_threshold:
                alert = {
                    "type": "DEPTH_ALERT",
                    "symbol": self.symbol,
                    "exchange": self.exchange,
                    "current_depth": current_depth,
                    "baseline_depth": self.baseline_depth,
                    "drop_pct": depth_drop,
                    "timestamp": orderbook.timestamp.isoformat(),
                    "message": f"Depth dropped by {depth_drop:.1f}%"
                }
                await self._trigger_alerts(alert)

    async def _trigger_alerts(self, alert: dict):
        """Trigger alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                print(f"Error in alert callback: {e}")

    async def start(self):
        """Start monitoring."""
        await self.stream.subscribe(self.symbol, self._handle_update)
        await self.stream.start()

    async def stop(self):
        """Stop monitoring."""
        await self.stream.stop()
