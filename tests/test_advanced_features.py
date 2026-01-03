"""
Tests for advanced features.

Covers:
- Multi-exchange comparison
- Historical tracking
- Alert system
- Market impact calculation
- Caching
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

# Import modules to test
from market_liquidity_monitor.agents.tools import compare_exchanges, calculate_market_impact
from market_liquidity_monitor.data_engine.models import (
    OrderBook,
    OrderBookLevel,
    HistoricalSnapshot,
    LiquidityAlert,
)
from market_liquidity_monitor.data_engine.historical import HistoricalTracker
from market_liquidity_monitor.data_engine.cache import CacheManager


# Fixtures
@pytest.fixture
def sample_orderbook():
    """Create sample order book for testing."""
    bids = [
        OrderBookLevel(price=100.0, amount=10.0),
        OrderBookLevel(price=99.5, amount=15.0),
        OrderBookLevel(price=99.0, amount=20.0),
    ]
    asks = [
        OrderBookLevel(price=100.5, amount=12.0),
        OrderBookLevel(price=101.0, amount=18.0),
        OrderBookLevel(price=101.5, amount=25.0),
    ]
    return OrderBook(
        symbol="TEST/USDT",
        exchange="test_exchange",
        bids=bids,
        asks=asks,
    )


@pytest.fixture
def sample_snapshot():
    """Create sample historical snapshot."""
    return HistoricalSnapshot(
        symbol="TEST/USDT",
        exchange="test_exchange",
        timestamp=datetime.utcnow(),
        best_bid=100.0,
        best_ask=100.5,
        spread=0.5,
        spread_percentage=0.5,
        mid_price=100.25,
        bid_volume_10=100.0,
        ask_volume_10=95.0,
        total_volume_20=195.0,
        liquidity_1pct_usd=10000.0,
        liquidity_2pct_usd=20000.0,
        imbalance_ratio=1.05,
    )


# Tests for Multi-Exchange Comparison
class TestMultiExchangeComparison:
    """Test multi-exchange comparison feature."""

    @pytest.mark.asyncio
    async def test_compare_exchanges_success(self):
        """Test successful multi-exchange comparison."""
        # Mock exchange manager
        with patch('market_liquidity_monitor.agents.tools.exchange_manager') as mock_manager:
            # Create mock client
            mock_client = AsyncMock()
            mock_client.fetch_order_book = AsyncMock(return_value=Mock(
                best_bid=Mock(price=100.0),
                best_ask=Mock(price=100.5),
                spread_percentage=0.5,
                get_cumulative_volume=Mock(return_value=50.0)
            ))

            mock_manager.get_client = AsyncMock(return_value=mock_client)

            # Run comparison
            result = await compare_exchanges(
                symbol="BTC/USDT",
                exchanges=["binance", "coinbase"],
                levels=20
            )

            # Assertions
            assert "exchanges_compared" in result
            assert len(result["exchanges_compared"]) == 2
            assert "best_bid_exchange" in result
            assert "best_ask_exchange" in result
            assert "tightest_spread_exchange" in result

    @pytest.mark.asyncio
    async def test_compare_exchanges_with_failure(self):
        """Test comparison when one exchange fails."""
        with patch('market_liquidity_monitor.agents.tools.exchange_manager') as mock_manager:
            # First exchange succeeds
            mock_client1 = AsyncMock()
            mock_client1.fetch_order_book = AsyncMock(return_value=Mock(
                best_bid=Mock(price=100.0),
                best_ask=Mock(price=100.5),
                spread_percentage=0.5,
                get_cumulative_volume=Mock(return_value=50.0)
            ))

            # Second exchange fails
            mock_client2 = AsyncMock()
            mock_client2.fetch_order_book = AsyncMock(side_effect=Exception("Network error"))

            async def get_client_mock(exchange):
                return mock_client1 if exchange == "binance" else mock_client2

            mock_manager.get_client = get_client_mock

            result = await compare_exchanges(
                symbol="BTC/USDT",
                exchanges=["binance", "coinbase"],
                levels=20
            )

            # Should have one successful fetch
            assert result["successful_fetches"] == 1
            assert len(result["failed_exchanges"]) == 1

    @pytest.mark.asyncio
    async def test_arbitrage_detection(self):
        """Test arbitrage opportunity detection."""
        with patch('market_liquidity_monitor.agents.tools.exchange_manager') as mock_manager:
            # Binance has lower ask
            mock_client_binance = AsyncMock()
            mock_client_binance.fetch_order_book = AsyncMock(return_value=Mock(
                best_bid=Mock(price=100.0),
                best_ask=Mock(price=100.3),  # Lower ask
                spread_percentage=0.3,
                get_cumulative_volume=Mock(return_value=50.0)
            ))

            # Coinbase has higher bid
            mock_client_coinbase = AsyncMock()
            mock_client_coinbase.fetch_order_book = AsyncMock(return_value=Mock(
                best_bid=Mock(price=100.8),  # Higher bid
                best_ask=Mock(price=101.0),
                spread_percentage=0.2,
                get_cumulative_volume=Mock(return_value=50.0)
            ))

            async def get_client_mock(exchange):
                return mock_client_binance if exchange == "binance" else mock_client_coinbase

            mock_manager.get_client = get_client_mock

            result = await compare_exchanges(
                symbol="BTC/USDT",
                exchanges=["binance", "coinbase"],
                levels=20
            )

            # Should detect arbitrage (buy on binance at 100.3, sell on coinbase at 100.8)
            assert result["arbitrage_opportunity_pct"] is not None
            assert result["arbitrage_opportunity_pct"] > 0
            assert "arbitrage_route" in result


# Tests for Market Impact Calculation
class TestMarketImpact:
    """Test market impact calculation."""

    @pytest.mark.asyncio
    async def test_calculate_market_impact_buy(self):
        """Test market impact for buy order."""
        with patch('market_liquidity_monitor.agents.tools.exchange_manager') as mock_manager:
            # Create order book with known liquidity
            mock_orderbook = Mock(
                asks=[
                    Mock(price=100.0, amount=10.0),  # $1000
                    Mock(price=100.5, amount=20.0),  # $2010
                    Mock(price=101.0, amount=30.0),  # $3030
                ],
                best_ask=Mock(price=100.0)
            )

            mock_client = AsyncMock()
            mock_client.fetch_order_book = AsyncMock(return_value=mock_orderbook)
            mock_manager.get_client = AsyncMock(return_value=mock_client)

            # Calculate impact for $2000 buy order
            result = await calculate_market_impact(
                symbol="BTC/USDT",
                order_size_usd=2000.0,
                side="buy",
                exchange="binance"
            )

            assert "sufficient_liquidity" in result
            assert result["sufficient_liquidity"] is True
            assert result["slippage_percentage"] >= 0  # Buy side = positive slippage
            assert result["levels_consumed"] >= 1

    @pytest.mark.asyncio
    async def test_calculate_market_impact_insufficient_liquidity(self):
        """Test impact calculation when liquidity is insufficient."""
        with patch('market_liquidity_monitor.agents.tools.exchange_manager') as mock_manager:
            mock_orderbook = Mock(
                asks=[Mock(price=100.0, amount=1.0)],  # Only $100 available
                best_ask=Mock(price=100.0)
            )

            mock_client = AsyncMock()
            mock_client.fetch_order_book = AsyncMock(return_value=mock_orderbook)
            mock_manager.get_client = AsyncMock(return_value=mock_client)

            result = await calculate_market_impact(
                symbol="BTC/USDT",
                order_size_usd=10000.0,  # Requesting $10k
                side="buy",
                exchange="binance"
            )

            assert "error" in result
            assert result["error"] == "Insufficient liquidity"


# Tests for Historical Tracking
class TestHistoricalTracking:
    """Test historical tracking system."""

    @pytest.mark.asyncio
    async def test_capture_snapshot(self, tmp_path):
        """Test snapshot capture."""
        tracker = HistoricalTracker(storage_dir=str(tmp_path))

        with patch('market_liquidity_monitor.data_engine.historical.exchange_manager') as mock_manager:
            # Mock order book
            mock_orderbook = Mock(
                best_bid=Mock(price=100.0),
                best_ask=Mock(price=100.5),
                spread=0.5,
                spread_percentage=0.5,
                get_cumulative_volume=Mock(return_value=100.0),
                get_liquidity_at_percentage=Mock(return_value=(50.0, 5000.0))
            )

            mock_client = AsyncMock()
            mock_client.fetch_order_book = AsyncMock(return_value=mock_orderbook)
            mock_manager.get_client = AsyncMock(return_value=mock_client)

            snapshot = await tracker.capture_snapshot("BTC/USDT", "binance")

            assert snapshot.symbol == "BTC/USDT"
            assert snapshot.exchange == "binance"
            assert snapshot.best_bid == 100.0
            assert snapshot.best_ask == 100.5

    @pytest.mark.asyncio
    async def test_baseline_calculation(self, tmp_path, sample_snapshot):
        """Test baseline metrics calculation."""
        tracker = HistoricalTracker(storage_dir=str(tmp_path))

        # Manually add snapshots to file
        import json
        file_path = tracker._get_snapshot_file("TEST/USDT", "test_exchange")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        snapshots = [sample_snapshot.model_dump(mode='json') for _ in range(10)]
        with open(file_path, 'w') as f:
            json.dump(snapshots, f)

        # Calculate baseline
        baseline = await tracker.get_baseline_metrics("TEST/USDT", "test_exchange", hours=24)

        assert "avg_spread_pct" in baseline
        assert "avg_volume" in baseline
        assert baseline["sample_count"] == 10

    @pytest.mark.asyncio
    async def test_anomaly_detection(self, tmp_path):
        """Test anomaly detection."""
        tracker = HistoricalTracker(storage_dir=str(tmp_path))

        with patch('market_liquidity_monitor.data_engine.historical.exchange_manager') as mock_manager:
            # Create baseline snapshots (normal spread)
            baseline_snapshots = []
            for i in range(10):
                snapshot = HistoricalSnapshot(
                    symbol="BTC/USDT",
                    exchange="binance",
                    timestamp=datetime.utcnow() - timedelta(hours=10-i),
                    best_bid=100.0,
                    best_ask=100.5,
                    spread=0.5,
                    spread_percentage=0.5,  # Normal 0.5%
                    mid_price=100.25,
                    bid_volume_10=100.0,
                    ask_volume_10=100.0,
                    total_volume_20=200.0,
                    liquidity_1pct_usd=10000.0,
                    liquidity_2pct_usd=20000.0,
                    imbalance_ratio=1.0,
                )
                baseline_snapshots.append(snapshot)

            # Store baseline
            import json
            file_path = tracker._get_snapshot_file("BTC/USDT", "binance")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump([s.model_dump(mode='json') for s in baseline_snapshots], f)

            # Mock current order book with WIDE spread (anomaly)
            mock_orderbook = Mock(
                best_bid=Mock(price=100.0),
                best_ask=Mock(price=102.0),  # 2% spread (4x normal)
                spread=2.0,
                spread_percentage=2.0,
                get_cumulative_volume=Mock(return_value=100.0),
                get_liquidity_at_percentage=Mock(return_value=(50.0, 5000.0))
            )

            mock_client = AsyncMock()
            mock_client.fetch_order_book = AsyncMock(return_value=mock_orderbook)
            mock_manager.get_client = AsyncMock(return_value=mock_client)

            # Detect anomalies
            alerts = await tracker.detect_anomalies("BTC/USDT", "binance", threshold_pct=30.0)

            # Should detect spread widening
            assert len(alerts) > 0
            spread_alerts = [a for a in alerts if a.alert_type == "SPREAD_WIDENING"]
            assert len(spread_alerts) > 0
            assert spread_alerts[0].severity in ["HIGH", "MEDIUM"]


# Tests for Caching
class TestCaching:
    """Test caching layer."""

    @pytest.mark.asyncio
    async def test_memory_cache_set_get(self):
        """Test in-memory cache set/get."""
        cache = CacheManager(use_redis=False)  # Force in-memory mode

        # Set value
        await cache.set("test_key", {"value": 123}, ttl=60)

        # Get value
        result = await cache.get("test_key")
        assert result == {"value": 123}

    @pytest.mark.asyncio
    async def test_cache_expiry(self):
        """Test cache TTL expiry."""
        cache = CacheManager(use_redis=False)

        # Set with 1 second TTL
        await cache.set("test_key", {"value": 123}, ttl=1)

        # Should exist immediately
        result = await cache.get("test_key")
        assert result is not None

        # Wait for expiry
        await asyncio.sleep(1.5)

        # Should be expired
        result = await cache.get("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_orderbook_cache(self, sample_orderbook):
        """Test order book caching."""
        cache = CacheManager(use_redis=False)

        # Cache orderbook
        await cache.cache_orderbook(
            symbol="BTC/USDT",
            exchange="binance",
            orderbook=sample_orderbook,
            ttl=10
        )

        # Retrieve
        cached = await cache.get_orderbook(
            symbol="BTC/USDT",
            exchange="binance"
        )

        assert cached is not None
        assert cached["symbol"] == "TEST/USDT"

    @pytest.mark.asyncio
    async def test_cache_invalidation(self):
        """Test cache invalidation."""
        cache = CacheManager(use_redis=False)

        # Set multiple keys
        await cache.set("mlm:orderbook:BTC_USDT:1", {"data": 1}, ttl=60)
        await cache.set("mlm:orderbook:BTC_USDT:2", {"data": 2}, ttl=60)
        await cache.set("mlm:comparison:BTC_USDT", {"data": 3}, ttl=60)

        # Clear pattern
        await cache.clear_pattern("mlm:orderbook:*")

        # Orderbook keys should be gone
        assert await cache.get("mlm:orderbook:BTC_USDT:1") is None
        assert await cache.get("mlm:orderbook:BTC_USDT:2") is None

        # Comparison key should remain
        assert await cache.get("mlm:comparison:BTC_USDT") is not None


# Integration test
class TestIntegration:
    """Integration tests combining multiple features."""

    @pytest.mark.asyncio
    async def test_comparison_with_caching(self):
        """Test that comparison results are cached."""
        cache = CacheManager(use_redis=False)

        with patch('market_liquidity_monitor.agents.tools.exchange_manager') as mock_manager:
            mock_client = AsyncMock()
            mock_client.fetch_order_book = AsyncMock(return_value=Mock(
                best_bid=Mock(price=100.0),
                best_ask=Mock(price=100.5),
                spread_percentage=0.5,
                get_cumulative_volume=Mock(return_value=50.0)
            ))
            mock_manager.get_client = AsyncMock(return_value=mock_client)

            # First call
            result1 = await compare_exchanges("BTC/USDT", ["binance", "coinbase"])

            # Cache it
            await cache.cache_comparison("BTC/USDT", ["binance", "coinbase"], result1)

            # Second call from cache
            result2 = await cache.get_comparison("BTC/USDT", ["binance", "coinbase"])

            assert result1 == result2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
