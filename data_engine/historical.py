"""
Historical liquidity tracking with time-series storage.

Provides snapshot capture, storage, and trend analysis for liquidity metrics.
"""

import asyncio
import json
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from collections import defaultdict

from .models import HistoricalSnapshot, OrderBook, LiquidityAlert
from . import exchange_manager


class HistoricalTracker:
    """
    Tracks and stores historical liquidity snapshots.

    Uses file-based JSON storage (can be upgraded to database later).
    """

    def __init__(self, storage_dir: str = "./data/historical"):
        """
        Initialize historical tracker.

        Args:
            storage_dir: Directory for storing snapshot files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, List[HistoricalSnapshot]] = defaultdict(list)

    def _get_snapshot_file(self, symbol: str, exchange: str) -> Path:
        """Get file path for symbol/exchange snapshots."""
        safe_symbol = symbol.replace("/", "_")
        return self.storage_dir / f"{exchange}_{safe_symbol}.json"

    async def capture_snapshot(
        self,
        symbol: str,
        exchange: str = "binance"
    ) -> HistoricalSnapshot:
        """
        Capture current liquidity snapshot.

        Args:
            symbol: Trading pair
            exchange: Exchange name

        Returns:
            HistoricalSnapshot with current metrics
        """
        # Fetch current order book
        client = await exchange_manager.get_client(exchange)
        orderbook = await client.fetch_order_book(symbol, limit=20)

        # Calculate metrics
        mid_price = (
            (orderbook.best_bid.price + orderbook.best_ask.price) / 2
            if orderbook.best_bid and orderbook.best_ask
            else 0
        )

        bid_volume = orderbook.get_cumulative_volume("bids", 10)
        ask_volume = orderbook.get_cumulative_volume("asks", 10)
        total_volume = bid_volume + ask_volume

        # Calculate liquidity in USD
        liq_1pct = orderbook.get_liquidity_at_percentage("bids", 1.0)
        liq_2pct = orderbook.get_liquidity_at_percentage("bids", 2.0)

        imbalance = bid_volume / ask_volume if ask_volume > 0 else 1.0

        snapshot = HistoricalSnapshot(
            symbol=symbol,
            exchange=exchange,
            timestamp=datetime.utcnow(),
            best_bid=orderbook.best_bid.price if orderbook.best_bid else 0,
            best_ask=orderbook.best_ask.price if orderbook.best_ask else 0,
            spread=orderbook.spread or 0,
            spread_percentage=orderbook.spread_percentage or 0,
            mid_price=mid_price,
            bid_volume_10=bid_volume,
            ask_volume_10=ask_volume,
            total_volume_20=total_volume,
            liquidity_1pct_usd=liq_1pct[1],
            liquidity_2pct_usd=liq_2pct[1],
            imbalance_ratio=imbalance,
        )

        # Store snapshot
        await self._store_snapshot(snapshot)

        return snapshot

    async def _store_snapshot(self, snapshot: HistoricalSnapshot):
        """
        Store snapshot to file.

        Args:
            snapshot: Snapshot to store
        """
        file_path = self._get_snapshot_file(snapshot.symbol, snapshot.exchange)

        # Load existing snapshots
        snapshots = []
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                snapshots = [HistoricalSnapshot(**s) for s in data]

        # Add new snapshot
        snapshots.append(snapshot)

        # Keep only last 1000 snapshots to prevent file bloat
        snapshots = snapshots[-1000:]

        # Save to file
        with open(file_path, 'w') as f:
            json.dump(
                [s.model_dump(mode='json') for s in snapshots],
                f,
                indent=2
            )

        # Update cache
        cache_key = f"{snapshot.exchange}:{snapshot.symbol}"
        self._cache[cache_key] = snapshots

    async def get_snapshots(
        self,
        symbol: str,
        exchange: str = "binance",
        hours: int = 24
    ) -> List[HistoricalSnapshot]:
        """
        Retrieve historical snapshots.

        Args:
            symbol: Trading pair
            exchange: Exchange name
            hours: Number of hours to retrieve

        Returns:
            List of historical snapshots
        """
        file_path = self._get_snapshot_file(symbol, exchange)

        if not file_path.exists():
            return []

        # Load from file
        with open(file_path, 'r') as f:
            data = json.load(f)
            snapshots = [HistoricalSnapshot(**s) for s in data]

        # Filter by time window
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        filtered = [
            s for s in snapshots
            if s.timestamp >= cutoff
        ]

        return filtered

    async def get_baseline_metrics(
        self,
        symbol: str,
        exchange: str = "binance",
        hours: int = 24
    ) -> Dict[str, float]:
        """
        Calculate baseline metrics from historical data.

        Args:
            symbol: Trading pair
            exchange: Exchange name
            hours: Lookback period in hours

        Returns:
            Dictionary with baseline metrics (avg, std, etc.)
        """
        snapshots = await self.get_snapshots(symbol, exchange, hours)

        if not snapshots:
            return {}

        # Calculate averages
        avg_spread = sum(s.spread_percentage for s in snapshots) / len(snapshots)
        avg_volume = sum(s.total_volume_20 for s in snapshots) / len(snapshots)
        avg_liquidity = sum(s.liquidity_1pct_usd for s in snapshots) / len(snapshots)
        avg_imbalance = sum(s.imbalance_ratio for s in snapshots) / len(snapshots)

        return {
            "avg_spread_pct": avg_spread,
            "avg_volume": avg_volume,
            "avg_liquidity_1pct_usd": avg_liquidity,
            "avg_imbalance_ratio": avg_imbalance,
            "sample_count": len(snapshots),
        }

    async def detect_anomalies(
        self,
        symbol: str,
        exchange: str = "binance",
        threshold_pct: float = 30.0
    ) -> List[LiquidityAlert]:
        """
        Detect anomalies in current liquidity vs baseline.

        Args:
            symbol: Trading pair
            exchange: Exchange name
            threshold_pct: Percentage deviation to trigger alert

        Returns:
            List of liquidity alerts
        """
        # Get current snapshot and baseline
        current = await self.capture_snapshot(symbol, exchange)
        baseline = await self.get_baseline_metrics(symbol, exchange, hours=24)

        if not baseline:
            return []

        alerts = []

        # Check spread widening
        if baseline.get("avg_spread_pct"):
            spread_deviation = (
                (current.spread_percentage - baseline["avg_spread_pct"])
                / baseline["avg_spread_pct"]
            ) * 100

            if abs(spread_deviation) > threshold_pct:
                severity = "HIGH" if abs(spread_deviation) > 50 else "MEDIUM"
                alerts.append(LiquidityAlert(
                    alert_id=f"{exchange}_{symbol}_spread_{current.timestamp.isoformat()}",
                    severity=severity,
                    symbol=symbol,
                    exchange=exchange,
                    alert_type="SPREAD_WIDENING",
                    current_value=current.spread_percentage,
                    baseline_value=baseline["avg_spread_pct"],
                    deviation_percentage=spread_deviation,
                    message=f"Spread widened by {spread_deviation:.1f}% (current: {current.spread_percentage:.3f}%, baseline: {baseline['avg_spread_pct']:.3f}%)",
                    suggested_action="Consider waiting for tighter spreads before trading",
                    requires_action=severity == "HIGH"
                ))

        # Check depth drop
        if baseline.get("avg_volume"):
            volume_deviation = (
                (current.total_volume_20 - baseline["avg_volume"])
                / baseline["avg_volume"]
            ) * 100

            if volume_deviation < -threshold_pct:
                severity = "HIGH" if volume_deviation < -50 else "MEDIUM"
                alerts.append(LiquidityAlert(
                    alert_id=f"{exchange}_{symbol}_depth_{current.timestamp.isoformat()}",
                    severity=severity,
                    symbol=symbol,
                    exchange=exchange,
                    alert_type="DEPTH_DROP",
                    current_value=current.total_volume_20,
                    baseline_value=baseline["avg_volume"],
                    deviation_percentage=volume_deviation,
                    message=f"Order book depth dropped by {abs(volume_deviation):.1f}% (current: {current.total_volume_20:.1f}, baseline: {baseline['avg_volume']:.1f})",
                    suggested_action="Reduce order size to minimize slippage",
                    requires_action=severity == "HIGH"
                ))

        # Check imbalance
        if baseline.get("avg_imbalance_ratio"):
            imbalance_deviation = (
                (current.imbalance_ratio - baseline["avg_imbalance_ratio"])
                / baseline["avg_imbalance_ratio"]
            ) * 100

            if abs(imbalance_deviation) > threshold_pct:
                pressure = "buying" if current.imbalance_ratio > 1 else "selling"
                severity = "MEDIUM"
                alerts.append(LiquidityAlert(
                    alert_id=f"{exchange}_{symbol}_imbalance_{current.timestamp.isoformat()}",
                    severity=severity,
                    symbol=symbol,
                    exchange=exchange,
                    alert_type="IMBALANCE",
                    current_value=current.imbalance_ratio,
                    baseline_value=baseline["avg_imbalance_ratio"],
                    deviation_percentage=imbalance_deviation,
                    message=f"Significant {pressure} pressure detected (imbalance ratio: {current.imbalance_ratio:.2f}, baseline: {baseline['avg_imbalance_ratio']:.2f})",
                    suggested_action=f"{'Price may increase' if pressure == 'buying' else 'Price may decrease'} - adjust entry strategy",
                    requires_action=False
                ))

        return alerts

    async def start_continuous_tracking(
        self,
        symbols: List[str],
        exchange: str = "binance",
        interval_seconds: int = 60
    ):
        """
        Start continuous background tracking.

        Args:
            symbols: List of trading pairs to track
            exchange: Exchange name
            interval_seconds: Snapshot interval in seconds
        """
        async def track_symbol(symbol: str):
            while True:
                try:
                    await self.capture_snapshot(symbol, exchange)
                    await asyncio.sleep(interval_seconds)
                except Exception as e:
                    print(f"Error tracking {symbol}: {e}")
                    await asyncio.sleep(interval_seconds)

        # Start tracking tasks for each symbol
        tasks = [track_symbol(symbol) for symbol in symbols]
        await asyncio.gather(*tasks)


# Global tracker instance
historical_tracker = HistoricalTracker()
