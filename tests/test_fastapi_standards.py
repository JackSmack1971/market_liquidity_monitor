import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app
from data_engine import exchange_manager
from unittest.mock import AsyncMock, patch
import asyncio
from datetime import datetime, timezone

@pytest.mark.asyncio
async def test_lifespan_preloads_markets():
    """Verify that the lifespan event pre-loads markets for priority exchanges."""
    # We patch the exchange_manager in api.main where it is used.
    with patch('api.main.exchange_manager.preload_exchange', AsyncMock()) as mock_preload:
        from api.main import lifespan
        # Manually drive the lifespan context to verify pre-loading
        async with lifespan(app):
            # Lifespan startup logic should have executed
            assert mock_preload.call_count >= 3
            # Check if it was called for binance, coinbase, kraken
            called_exchanges = [call.args[0] for call in mock_preload.call_args_list]
            assert "binance" in called_exchanges
            assert "coinbase" in called_exchanges
            assert "kraken" in called_exchanges

@pytest.mark.asyncio
async def test_orderbook_yield_dependency_and_background_task():
    """Verify that get_orderbook uses yield dependency and triggers background snapshot."""
    # Mock capture_snapshot on the global tracker instance
    with patch('data_engine.historical.historical_tracker.capture_snapshot', AsyncMock()) as mock_capture:
        # Mock ExchangeClient.fetch_order_book to avoid real network calls
        with patch('api.dependencies.exchange_manager.get_client', AsyncMock()) as mock_get_client:
            mock_client = AsyncMock()
            # Set up as an async context manager
            mock_client.__aenter__.return_value = mock_client
            
            # Create a valid OrderBook dictionary for the mock to return
            valid_orderbook = {
                "symbol": "BTCUSDT",
                "exchange": "binance",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "bids": [{"price": 99.0, "amount": 1.0}],
                "asks": [{"price": 101.0, "amount": 1.0}],
                "best_bid": {"price": 99.0, "amount": 1.0},
                "best_ask": {"price": 101.0, "amount": 1.0},
                "spread": 2.0,
                "spread_percentage": 2.0,
                "bid_depth_10": 1.0,
                "ask_depth_10": 1.0
            }
            mock_client.fetch_order_book.return_value = valid_orderbook
            mock_get_client.return_value = mock_client
            
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                # Use a symbol without a slash to avoid path complexity in test
                response = await ac.get("/api/v1/orderbook/BTCUSDT?exchange=binance")
                
                assert response.status_code == 200
                await asyncio.sleep(0.1) 
                # Verify background task was called
                mock_capture.assert_called()

@pytest.mark.asyncio
async def test_analyze_background_task():
    """Verify that /analyze triggers background snapshot when symbol/exchange are provided."""
    with patch('data_engine.historical.historical_tracker.capture_snapshot', AsyncMock()) as mock_capture:
        with patch('api.routes.MarketAnalyzer.analyze_liquidity', AsyncMock()) as mock_analyze:
            # Provide valid LiquidityScorecard data to pass validation
            valid_analysis = {
                "symbol": "SOL/USDT",
                "exchange": "binance",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "liquidity_score": 8,
                "estimated_slippage_percent": 0.05,
                "recommended_max_size": 10000.0,
                "risk_factors": ["None"],
                "summary_analysis": "Test scorecard summary",
                "spread_pct": 0.01,
                "bid_depth_10": 500.0,
                "ask_depth_10": 500.0,
                "volatility_rating": "STABLE"
            }
            mock_analyze.return_value = valid_analysis
            
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                payload = {
                    "query": "Analysis for SOL",
                    "symbol": "SOL/USDT",
                    "exchange": "binance"
                }
                response = await ac.post("/api/v1/analyze", json=payload)
                
                assert response.status_code == 200
                await asyncio.sleep(0.1)
                mock_capture.assert_called()
