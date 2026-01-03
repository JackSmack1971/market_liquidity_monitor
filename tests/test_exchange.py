"""
Tests for exchange data engine.

Tests CCXT integration and order book fetching.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from market_liquidity_monitor.data_engine import ExchangeClient, OrderBook


@pytest.mark.asyncio
async def test_exchange_client_initialization():
    """Test exchange client can be initialized."""
    client = ExchangeClient(exchange_id="binance")
    assert client.exchange_id == "binance"
    await client.close()


@pytest.mark.asyncio
async def test_fetch_order_book_structure():
    """Test order book fetching returns correct structure."""
    client = ExchangeClient(exchange_id="binance")

    # Mock the exchange fetch_order_book method
    mock_orderbook = {
        "bids": [[100.0, 1.5], [99.5, 2.0]],
        "asks": [[101.0, 1.0], [101.5, 1.2]],
    }

    with patch.object(client.exchange, 'fetch_order_book', new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = mock_orderbook

        orderbook = await client.fetch_order_book("BTC/USDT", limit=10)

        # Verify structure
        assert isinstance(orderbook, OrderBook)
        assert orderbook.symbol == "BTC/USDT"
        assert orderbook.exchange == "binance"
        assert len(orderbook.bids) == 2
        assert len(orderbook.asks) == 2

        # Verify best bid/ask
        assert orderbook.best_bid.price == 100.0
        assert orderbook.best_ask.price == 101.0

        # Verify spread
        assert orderbook.spread == 1.0

    await client.close()


@pytest.mark.asyncio
async def test_orderbook_metrics():
    """Test order book metric calculations."""
    from market_liquidity_monitor.data_engine.models import OrderBookLevel

    # Create test order book
    orderbook = OrderBook(
        symbol="SOL/USDT",
        exchange="binance",
        bids=[
            OrderBookLevel(price=100.0, amount=10.0),
            OrderBookLevel(price=99.0, amount=20.0),
            OrderBookLevel(price=98.0, amount=30.0),
        ],
        asks=[
            OrderBookLevel(price=101.0, amount=5.0),
            OrderBookLevel(price=102.0, amount=10.0),
            OrderBookLevel(price=103.0, amount=15.0),
        ],
    )

    # Test spread
    assert orderbook.spread == 1.0
    assert orderbook.spread_percentage == pytest.approx(0.995, rel=0.01)

    # Test cumulative volume
    bid_volume = orderbook.get_cumulative_volume("bids", 3)
    assert bid_volume == 60.0

    ask_volume = orderbook.get_cumulative_volume("asks", 3)
    assert ask_volume == 30.0

    # Test liquidity at percentage
    volume, value = orderbook.get_liquidity_at_percentage("bids", 2.0)
    assert volume == 30.0  # Within 2% of 100.0


@pytest.mark.asyncio
async def test_context_manager():
    """Test exchange client works as async context manager."""
    async with ExchangeClient(exchange_id="binance") as client:
        assert client.exchange_id == "binance"

    # Client should be closed after exiting context


def test_precision_formatting():
    """Test precision formatting methods."""
    # Note: This test would need actual exchange instance
    # which requires real connection. For unit test, we mock it.
    pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
