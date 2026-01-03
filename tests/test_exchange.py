import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from datetime import datetime
import ccxt.async_support as ccxt
from data_engine.exchange import ExchangeClient, ExchangeManager, with_retry
from data_engine.models import OrderBook, OrderBookLevel
from data_engine.circuit_breaker import CircuitBreaker

class TestExchangeIntegration(unittest.IsolatedAsyncioTestCase):
    """Tests for the Exchange Integration layer."""

    async def asyncSetUp(self):
        """Setup mock environment."""
        self.exchange_id = "binance"
        self.symbol = "BTC/USDT"
        
        # Patch ccxt.binance to return a mock class
        self.mock_ccxt_instance = MagicMock()
        self.mock_ccxt_instance.load_markets = AsyncMock()
        self.mock_ccxt_instance.fetch_order_book = AsyncMock()
        self.mock_ccxt_instance.fetch_ohlcv = AsyncMock()
        self.mock_ccxt_instance.close = AsyncMock()
        self.mock_ccxt_instance.markets = {}
        self.mock_ccxt_instance.has = {"fetchOHLCV": True}
        
        self.patcher_ccxt = patch(f'ccxt.async_support.{self.exchange_id}', return_value=self.mock_ccxt_instance)
        self.patcher_ccxt.start()
        
        # Patch settings
        self.patcher_settings = patch('data_engine.exchange.settings')
        self.mock_settings = self.patcher_settings.start()
        self.mock_settings.default_exchange = "binance"
        self.mock_settings.exchange_api_key = None
        self.mock_settings.exchange_api_secret = None
        self.mock_settings.logfire_token = None

        self.client = ExchangeClient(self.exchange_id)

    async def asyncTearDown(self):
        """Cleanup."""
        self.patcher_ccxt.stop()
        self.patcher_settings.stop()

    # ========== Retry Decorator Tests ==========

    async def test_with_retry_rate_limit(self):
        """Test with_retry decorator handles RateLimitExceeded."""
        mock_self = MagicMock()
        mock_self.exchange_id = "binance"
        
        call_count = 0
        async def failing_func(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ccxt.RateLimitExceeded("Slow down")
            return "success"

        decorated = with_retry(retries=2)(failing_func)
        
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            result = await decorated(mock_self)
            self.assertEqual(result, "success")
            self.assertEqual(call_count, 2)
            mock_sleep.assert_awaited_with(10.0)

    async def test_client_lifecycle_and_status(self):
        """Test status property and close."""
        self.client.circuit_breaker._failures = 2
        status = self.client.status
        self.assertEqual(status["failures"], 2)
        self.assertEqual(status["state"], "CLOSED")
        
        await self.client.close()
        self.mock_ccxt_instance.close.assert_awaited_once()

    async def test_search_symbol(self):
        """Test search_symbol filters markets."""
        self.client._markets_loaded = True
        self.client.exchange.markets = {
            "BTC/USDT": {"symbol": "BTC/USDT"},
            "SOL/USDT": {"symbol": "SOL/USDT"},
            "BTC/FDUSD": {"symbol": "BTC/FDUSD"}
        }
        
        results = await self.client.search_symbol("BTC")
        self.assertEqual(len(results), 2)
        self.assertIn("BTC/USDT", results)
        self.assertIn("BTC/FDUSD", results)

    async def test_with_retry_network_error(self):
        """Test with_retry decorator handles NetworkError with backoff."""
        mock_self = MagicMock()
        mock_self.exchange_id = "binance"
        
        call_count = 0
        async def failing_func(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ccxt.NetworkError("Timeout")
            return "success"

        decorated = with_retry(retries=3, backoff=0.1)(failing_func)
        
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            result = await decorated(mock_self)
            self.assertEqual(result, "success")
            self.assertEqual(call_count, 3)
            # Two sleeps: 0.1, 0.2
            self.assertEqual(mock_sleep.call_count, 2)

    def test_client_init_with_keys(self):
        """Test initialization with explicit API keys."""
        client = ExchangeClient(self.exchange_id, api_key="key", api_secret="secret")
        self.assertEqual(client.api_key, "key")
        # Verify passed to CCXT via config
        # We can check if the mock was initialized with config containing keys
        # But for now, client instance check is sufficient validation of our logic
        self.assertEqual(client.api_secret, "secret")

    async def test_fetch_order_book_fee_fallback(self):
        """Test fetch_order_book uses market data for fees if exchange.fees missing."""
        self.mock_ccxt_instance.fetch_order_book.return_value = {
            "bids": [], "asks": [], "timestamp": 1700000000000
        }
        # Clear global fees, set specific market fee
        del self.mock_ccxt_instance.fees
        self.mock_ccxt_instance.markets = {self.symbol: {'taker': 0.003}}
        
        self.client._markets_loaded = True
        orderbook = await self.client.fetch_order_book(self.symbol)
        self.assertEqual(orderbook.taker_fee_pct, 0.3)

    async def test_fetch_markets(self):
        """Test fetch_markets wrapper."""
        # Ensure it behaves as AsyncMock
        self.mock_ccxt_instance.fetch_markets = AsyncMock(return_value=[{"symbol": "BTC/USDT"}])
        markets = await self.client.fetch_markets()
        self.assertEqual(len(markets), 1)
    
    def test_cost_to_precision(self):
        """Test cost_to_precision wrapper."""
        self.client.exchange.cost_to_precision.return_value = "10.50"
        self.assertEqual(self.client.cost_to_precision(self.symbol, 10.501), "10.50")

    async def test_with_retry_generic_exception(self):
        """Test retry decorator re-raises generic exceptions immediately."""
        async def fail_gen(self):
            raise ValueError("Generic Error")
            
        decorated = with_retry()(fail_gen)
        
        with self.assertRaises(ValueError):
            await decorated(self.client)

    # ========== ExchangeClient Tests ==========

    def test_client_initialization(self):
        """Test ExchangeClient init and status."""
        self.assertEqual(self.client.exchange_id, "binance")
        self.assertTrue(self.client.status["is_healthy"])
        self.assertEqual(self.client.status["state"], "CLOSED")

    async def test_fetch_order_book_success(self):
        """Test fetch_order_book model conversion and fees."""
        self.mock_ccxt_instance.fetch_order_book.return_value = {
            "bids": [[100.0, 1.0]],
            "asks": [[101.0, 2.0]],
            "timestamp": 1700000000000
        }
        self.client.exchange.fees = {'trading': {'taker': 0.002}} # 0.2%
        
        # Bypass ensure_markets_loaded
        self.client._markets_loaded = True
        
        orderbook = await self.client.fetch_order_book(self.symbol)
        
        self.assertIsInstance(orderbook, OrderBook)
        self.assertEqual(orderbook.taker_fee_pct, 0.2)
        self.assertEqual(len(orderbook.bids), 1)
        self.assertEqual(orderbook.bids[0].price, 100.0)
        self.assertIsNotNone(orderbook.latency_ms)

    async def test_fetch_ohlcv_caching(self):
        """Test fetch_ohlcv interactions with cache."""
        candles = [[1700000000000, 10, 11, 9, 10.5, 100]]
        self.mock_ccxt_instance.fetch_ohlcv.return_value = candles
        self.client._markets_loaded = True
        
        with patch('data_engine.cache.cache_manager.get', new_callable=AsyncMock) as mock_cache_get, \
             patch('data_engine.cache.cache_manager.set', new_callable=AsyncMock) as mock_cache_set:
            
            # 1. Miss cache
            mock_cache_get.return_value = None
            result = await self.client.fetch_ohlcv(self.symbol)
            self.assertEqual(result, candles)
            mock_cache_set.assert_awaited()
            
            # 2. Hit cache
            mock_cache_get.return_value = candles
            result2 = await self.client.fetch_ohlcv(self.symbol)
            self.assertEqual(result2, candles)
            # Should not call fetch_ohlcv again
            self.mock_ccxt_instance.fetch_ohlcv.assert_called_once()

    def test_validate_order_limits(self):
        """Test order limit validation logic."""
        self.client._markets_loaded = True
        self.client.exchange.market = MagicMock(return_value={
            'limits': {
                'amount': {'min': 0.1, 'max': 100.0},
                'cost': {'min': 10.0}
            }
        })
        
        # Valid
        ok, msg = self.client.validate_order_limits(self.symbol, 1.0, 50.0)
        self.assertTrue(ok)
        
        # Below min amount
        ok, msg = self.client.validate_order_limits(self.symbol, 0.05, 50.0)
        self.assertFalse(ok)
        self.assertIn("below the minimum required", msg)
        
        # Above max amount
        ok, msg = self.client.validate_order_limits(self.symbol, 150.0, 10.0)
        self.assertFalse(ok)
        self.assertIn("exceeds the maximum allowed", msg)
        
        # Below min cost (0.1 * 50 = 5.0 < 10.0)
        ok, msg = self.client.validate_order_limits(self.symbol, 0.1, 50.0)
        self.assertFalse(ok)
        self.assertIn("cost $5.00 is below the minimum", msg)

        # Symbol not found
        self.client.exchange.market.side_effect = Exception("Not found")
        ok, msg = self.client.validate_order_limits("FAKE/USD", 1.0, 1.0)
        self.assertFalse(ok)
        self.assertIn("not found", msg)

    def test_precision_wrappers(self):
        """Test CCXT precision wrappers."""
        self.client.exchange.amount_to_precision.return_value = "1.234"
        self.assertEqual(self.client.amount_to_precision("BTC/USDT", 1.23456), "1.234")
        
        self.client.exchange.price_to_precision.return_value = "50000.1"
        self.assertEqual(self.client.price_to_precision("BTC/USDT", 50000.123), "50000.1")

    async def test_get_market_info(self):
        """Test get_market_info with cache."""
        self.client._markets_loaded = True
        self.client.exchange.markets = {"BTC/USDT": {"id": "btc_usdt"}}
        info = await self.client.get_market_info("BTC/USDT")
        self.assertEqual(info["id"], "btc_usdt")
        
        with self.assertRaises(ValueError):
            await self.client.get_market_info("UNKNOWN")

    # ========== ExchangeManager Tests ==========

    async def test_exchange_manager_pooling(self):
        """Test ExchangeManager pools clients and handles dynamic credentials."""
        manager = ExchangeManager()
        
        # 1. Get client (initializes)
        client1 = await manager.get_client("binance")
        self.assertIn("binance", manager._clients)
        
        # 2. Get same client (pooled)
        client2 = await manager.get_client("binance")
        self.assertEqual(client1, client2)
        
        # 3. Dynamic credential update
        with patch.object(client1.exchange, 'apiKey', create=True) as mock_key, \
             patch.object(client1.exchange, 'secret', create=True) as mock_secret:
            await manager.get_client("binance", api_key="new_key", api_secret="new_secret")
            self.assertEqual(client1.api_key, "new_key")
            self.assertEqual(client1.api_secret, "new_secret")
            self.assertEqual(client1.exchange.apiKey, "new_key")
            self.assertEqual(client1.exchange.secret, "new_secret")
            
        await manager.close_all()
        self.assertEqual(len(manager._clients), 0)

    async def test_preload_exchange(self):
        """Test pre-loading markets warms the manager cache."""
        manager = ExchangeManager()
        # Mock ExchangeClient within preload
        with patch('data_engine.exchange.ExchangeClient', new_callable=MagicMock) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.exchange.markets = {"BTC/USDT": {}}
            mock_client_class.return_value = mock_client
            
            await manager.preload_exchange("binance")
            self.assertIn("binance", manager._markets_cache)
            self.assertEqual(manager._markets_cache["binance"], {"BTC/USDT": {}})

if __name__ == '__main__':
    unittest.main()
