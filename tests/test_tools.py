import unittest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pydantic_ai import ModelRetry, RunContext
from agents.tools import (
    get_order_book_depth, 
    calculate_market_impact, 
    compare_exchanges,
    get_market_metadata,
    search_trading_pairs,
    run_historical_backtest,
    get_historical_metrics
)
from data_engine.models import OrderBook, MarketImpactReport, CrossExchangeComparison, BacktestReport

class TestAgentTools(unittest.IsolatedAsyncioTestCase):
    """Comprehensive tests for Agent Tools."""

    async def asyncSetUp(self):
        """Setup mock context."""
        self.ctx = MagicMock(spec=RunContext)
        
        # Mock clients
        self.mock_client = AsyncMock()
        self.mock_client.exchange = MagicMock() # Regular mock for sync properties by default
        self.mock_client.exchange_id = "binance"
        self.mock_client.circuit_breaker.state = "CLOSED"
        self.mock_client.exchange.has = {"fetchOHLCV": True, "fetchOrderBook": True}
        self.mock_client.exchange.milliseconds.return_value = 1700000000000
        
        # Async methods
        self.mock_client.fetch_order_book = AsyncMock()
        self.mock_client.fetch_ohlcv = AsyncMock()
        self.mock_client.exchange.fetch_order_book = AsyncMock()
        self.mock_client.exchange.fetch_ohlcv = AsyncMock()
        self.mock_client.exchange.sleep = AsyncMock()
        
        # Mock exchange_manager
        self.mock_client.exchange.market = MagicMock(return_value={"limits": {"amount": {"min": 0.1}, "cost": {"min": 10.0}}})
        self.patcher_manager = patch('agents.tools.exchange_manager.get_client', return_value=self.mock_client)
        self.mock_get_client = self.patcher_manager.start()

    async def asyncTearDown(self):
        self.patcher_manager.stop()

    # ========== HAPPY PATH TESTS ==========

    async def test_get_order_book_depth_happy_path(self):
        """Test get_order_book_depth with valid input."""
        mock_ob = MagicMock(spec=OrderBook)
        self.mock_client.fetch_order_book.return_value = mock_ob
        
        result = await get_order_book_depth(self.ctx, symbol="BTC/USDT", exchange="binance")
        
        self.assertEqual(result, mock_ob)
        self.mock_client.fetch_order_book.assert_called_with("BTC/USDT", limit=20)

    async def test_calculate_market_impact_happy_path(self):
        """Test calculate_market_impact simulation success."""
        # 1. Mock exchange and market info
        self.mock_client.exchange.market.return_value = {"limits": {"amount": {"min": 0.1}}}
        self.mock_client.exchange.amount_to_precision.return_value = "1.5"
        
        # 2. Mock orderbook and analytics
        mock_ob = MagicMock(spec=OrderBook)
        self.mock_client.fetch_order_book.return_value = mock_ob
        
        mock_report = MagicMock(spec=MarketImpactReport)
        mock_report.warning = None
        
        with patch('agents.tools.analytics.calculate_market_impact', return_value=mock_report) as mock_calc:
            result = await calculate_market_impact(self.ctx, symbol="BTC/USDT", side="buy", size=1.5)
            
            self.assertEqual(result, mock_report)
            mock_calc.assert_called()

    async def test_compare_exchanges_basic_happy_path(self):
        """Test basic compare_exchanges (order_size=0)."""
        mock_ob = MagicMock(spec=OrderBook)
        mock_ob.best_bid.price = 100
        mock_ob.best_ask.price = 101
        mock_ob.spread_percentage = 1.0
        mock_ob.get_cumulative_volume.return_value = 500
        mock_ob.latency_ms = 100
        mock_ob.circuit_state = "CLOSED"
        
        self.mock_client.fetch_order_book.return_value = mock_ob
        
        result = await compare_exchanges(self.ctx, symbol="BTC/USDT", exchanges=["binance"])
        
        self.assertEqual(result["best_bid_price"], 100)
        self.assertEqual(result["successful_fetches"], 1)

    # ========== EDGE CASE TESTS ==========

    async def test_get_order_book_depth_edge_open_circuit(self):
        """Test get_order_book_depth triggers ModelRetry on OPEN circuit."""
        self.mock_client.circuit_breaker.state = "OPEN"
        
        with self.assertRaises(ModelRetry) as cm:
            await get_order_book_depth(self.ctx, symbol="BTC/USDT")
        
        self.assertIn("suspended", str(cm.exception))

    async def test_calculate_market_impact_edge_precision_fail(self):
        """Test precision enforcement handles failure gracefully."""
        self.mock_client.exchange.amount_to_precision.side_effect = Exception("err")
        mock_ob = MagicMock(spec=OrderBook)
        self.mock_client.fetch_order_book.return_value = mock_ob
        
        with patch('agents.tools.analytics.calculate_market_impact', return_value=MagicMock(warning=None)):
            # Should not raise exception
            await calculate_market_impact(self.ctx, symbol="BTC/USDT", side="buy", size=1.0)

    # ========== ERROR SCENARIO TESTS ==========

    async def test_get_order_book_depth_raises_ModelRetry_symbol_not_found(self):
        """Test raises ModelRetry when symbol not found."""
        self.mock_client.fetch_order_book.side_effect = ValueError("Symbol not found")
        
        with self.assertRaises(ModelRetry) as cm:
            await get_order_book_depth(self.ctx, symbol="WRONG", exchange="binance")
        
        self.assertIn("search_trading_pairs", str(cm.exception))

    async def test_calculate_market_impact_raises_ModelRetry_limit_violation(self):
        """Test triggers ModelRetry for minimum size violation."""
        mock_report = MagicMock(spec=MarketImpactReport)
        mock_report.warning = "below exchange minimum"
        
        mock_ob = MagicMock(spec=OrderBook)
        self.mock_client.fetch_order_book.return_value = mock_ob
        
        with patch('agents.tools.analytics.calculate_market_impact', return_value=mock_report):
            with self.assertRaises(ModelRetry) as cm:
                await calculate_market_impact(self.ctx, symbol="BTC/USDT", side="buy", size=0.0001)
            
            self.assertIn("validation failed", str(cm.exception).lower())

    # ========== NEW TESTS FOR COVERAGE ==========

    async def test_search_trading_pairs_happy_path(self):
        """Test search_trading_pairs returns symbols."""
        self.mock_client.search_symbol.return_value = ["BTC/USDT", "BTC/USD"]
        result = await search_trading_pairs(self.ctx, query="BTC")
        self.assertEqual(result, ["BTC/USDT", "BTC/USD"])

    async def test_get_market_metadata_happy_path(self):
        """Test get_market_metadata returns expected dict."""
        self.mock_client.get_market_info.return_value = {
            "symbol": "BTC/USDT", "base": "BTC", "quote": "USDT",
            "active": True, "precision": {}, "limits": {}
        }
        self.mock_client.last_request_latency_ms = 50
        result = await get_market_metadata(self.ctx, symbol="BTC/USDT")
        self.assertEqual(result["symbol"], "BTC/USDT")
        self.assertEqual(result["latency_ms"], 50)

    async def test_compare_exchanges_advanced_happy_path(self):
        """Test compare_exchanges with order_size > 0."""
        mock_comparison = MagicMock(spec=CrossExchangeComparison)
        with patch('agents.tools.analytics.compare_liquidity_across_venues', return_value=mock_comparison):
            result = await compare_exchanges(self.ctx, symbol="BTC/USDT", order_size=100.0)
            self.assertEqual(result, mock_comparison)

    async def test_run_historical_backtest_happy_path(self):
        """Test run_historical_backtest success path."""
        self.mock_client.exchange.has = {'fetchOHLCV': True}
        self.mock_client.exchange.fetch_ohlcv = AsyncMock(return_value=[[1700000000000, 10, 11, 9, 10.5, 1000]])
        self.mock_client.exchange.market = MagicMock(return_value={"limits": {}})
        self.mock_client.exchange.milliseconds.return_value = 1700000000000
        
        async def mock_call(f, *args, **kwargs):
            return await f(*args, **kwargs)
        self.mock_client.circuit_breaker.call = AsyncMock(side_effect=mock_call)
        
        mock_report = MagicMock(spec=BacktestReport)
        mock_report.avg_slippage_bps = 5.0
        mock_report.max_slippage_bps = 10.0
        mock_report.high_risk_periods = 0
        mock_report.optimal_execution_windows = []
        mock_report.total_candles = 1
        
        with patch('agents.tools.analytics.simulate_historical_execution', return_value=mock_report):
            result = await run_historical_backtest(self.ctx, "BTC/USDT", 1.0, "buy", "1h")
            self.assertEqual(result, mock_report)

    async def test_run_historical_backtest_rate_limit_retry(self):
        """Test run_historical_backtest handles rate limit retry."""
        import ccxt
        self.mock_client.exchange.has = {'fetchOHLCV': True}
        # Fail once with rate limit, then succeed
        self.mock_client.exchange.fetch_ohlcv = AsyncMock(side_effect=[
            ccxt.RateLimitExceeded("too fast"),
            [[1700000000000, 10, 11, 9, 10.5, 1000]]
        ])
        self.mock_client.exchange.sleep = AsyncMock()
        self.mock_client.exchange.milliseconds.return_value = 1700000000000
        
        async def mock_call(f, *args, **kwargs):
            return await f(*args, **kwargs)
        self.mock_client.circuit_breaker.call = AsyncMock(side_effect=mock_call)
        
        mock_report = MagicMock()
        mock_report.avg_slippage_bps = 0.0
        mock_report.max_slippage_bps = 0.0
        mock_report.high_risk_periods = 0
        mock_report.optimal_execution_windows = []
        mock_report.total_candles = 1
        
        with patch('agents.tools.analytics.simulate_historical_execution', return_value=mock_report):
             await run_historical_backtest(self.ctx, "BTC/USDT", 1.0, "buy", "1h")
             # Verify sleep was called once
             self.mock_client.exchange.sleep.assert_called_once()

    async def test_get_historical_metrics_happy_path(self):
        """Test get_historical_metrics success path."""
        ohlcv_data = [
            [1700000000000, 10, 11, 9, 10, 1000],
            [1700000060000, 10, 12, 10, 11, 2000]
        ]
        self.mock_client.fetch_ohlcv.return_value = ohlcv_data
        self.mock_client.exchange.fetch_ohlcv.return_value = ohlcv_data
        
        result = await get_historical_metrics(self.ctx, symbol="BTC/USDT")
        self.assertEqual(result["candle_count"], 2)
        self.assertEqual(result["avg_volume"], 1500.0)

    async def test_get_order_book_depth_exchange_not_found(self):
        """Test get_order_book_depth raises ModelRetry for missing exchange."""
        # Force a generic exception with "exchange not found" string
        self.mock_get_client.side_effect = Exception("Exchange not found")
        
        with self.assertRaises(ModelRetry) as cm:
            await get_order_book_depth(self.ctx, symbol="BTC/USDT", exchange="unknown")
        
        self.assertIn("not supported", str(cm.exception))
        self.mock_get_client.side_effect = None # Reset for other tests

    async def test_run_historical_backtest_ohlcv_not_supported(self):
        """Test run_historical_backtest raises ModelRetry when OHLCV not supported."""
        self.mock_client.exchange.has = {'fetchOHLCV': False}
        with self.assertRaises(ModelRetry) as cm:
            await run_historical_backtest(self.ctx, "BTC/USDT", 1.0, "buy", "1h")
        self.assertIn("does not support historical OHLCV", str(cm.exception))

    async def test_run_historical_backtest_no_data(self):
        """Test run_historical_backtest raises ModelRetry when no data found."""
        self.mock_client.exchange.has = {'fetchOHLCV': True}
        self.mock_client.exchange.fetch_ohlcv = AsyncMock(return_value=[])
        
        async def mock_call(f, *args, **kwargs):
            return await f(*args, **kwargs)
        self.mock_client.circuit_breaker.call = AsyncMock(side_effect=mock_call)
        
        with self.assertRaises(ModelRetry) as cm:
            await run_historical_backtest(self.ctx, "BTC/USDT", 1.0, "buy", "1h")
        self.assertIn("No historical data available", str(cm.exception))

if __name__ == '__main__':
    unittest.main()
