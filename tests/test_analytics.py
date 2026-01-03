import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime
from data_engine.analytics import (
    calculate_market_impact,
    compare_liquidity_across_venues,
    reconstruct_synthetic_book,
    simulate_historical_execution
)
from data_engine.models import OrderBook, OrderBookLevel, MarketImpactReport, CrossExchangeComparison

class TestAnalytics(unittest.IsolatedAsyncioTestCase):
    """Tests for the Analytics Engine."""

    def setUp(self):
        """Build a dummy order book."""
        self.symbol = "BTC/USDT"
        self.bids = [
            OrderBookLevel(price=100.0, amount=1.0),
            OrderBookLevel(price=99.0, amount=2.0),
            OrderBookLevel(price=98.0, amount=5.0)
        ]
        self.asks = [
            OrderBookLevel(price=101.0, amount=1.0),
            OrderBookLevel(price=102.0, amount=2.0),
            OrderBookLevel(price=103.0, amount=5.0)
        ]
        self.orderbook = OrderBook(
            symbol=self.symbol,
            exchange="binance",
            bids=self.bids,
            asks=self.asks,
            timestamp=datetime.now(),
            latency_ms=50
        )

    # ========== Market Impact Tests ==========

    def test_calculate_market_impact_buy_base(self):
        """Test buy order with base size (SOL)."""
        # Buy 1.5 BTC.
        # 1.0 @ 101.0
        # 0.5 @ 102.0
        # Total cost: 101*1 + 102*0.5 = 101 + 51 = 152
        # VWAP: 152 / 1.5 = 101.3333
        report = calculate_market_impact(self.orderbook, side="buy", size=1.5, is_quote_size=False)
        
        self.assertEqual(report.target_size, 1.5)
        self.assertAlmostEqual(report.expected_fill_price, 101.33333333)
        self.assertEqual(report.reference_price, 101.0)
        self.assertTrue(report.slippage_bps > 0)

    def test_calculate_market_impact_sell_quote(self):
        """Test sell order with quote size (USD)."""
        # Sell 150 USD.
        # 1 BTC @ 100.0 = 100 USD. Remaining: 50 USD.
        # 0.5 BTC @ 99.0 = 49.5 USD. Remaining: 0.5 USD.
        # Wait, the walk adds cost until target USD.
        # 1.0 @ 100 = 100
        # 0.505 @ 99 = 49.995... approx 150.
        report = calculate_market_impact(self.orderbook, side="sell", size=150.0, is_quote_size=True)
        
        self.assertAlmostEqual(report.target_value_usd, 150.0)
        self.assertTrue(report.expected_fill_price < 100.0)

    def test_calculate_market_impact_quote_full_fill(self):
        """Test quote size fill where levels are partially consumed."""
        # Buy 150 USD.
        # 1.0 @ 101.0 = 101 USD. Remaining: 49 USD.
        # 2.0 @ 102.0 = 204 USD available.
        # Need 49 USD from 204 USD.
        # Fill = 49 / 102 = 0.48039 BTC.
        # Total BTC = 1.48039.
        report = calculate_market_impact(self.orderbook, side="buy", size=150.0, is_quote_size=True)
        self.assertAlmostEqual(report.target_value_usd, 150.0)
        self.assertAlmostEqual(report.target_size, 1.4803921568, places=6)

    def test_calculate_market_impact_sell_base_exhaust(self):
        """Test sell order exceeding available volume."""
        # Book has 1+2+5 = 8 BTC. Sell 10 BTC.
        report = calculate_market_impact(self.orderbook, side="sell", size=10.0, is_quote_size=False)
        self.assertIn("exceeds available book depth", report.warning)
        self.assertEqual(report.target_size, 8.0)

    def test_calculate_market_impact_empty_book(self):
        """Test handling of empty order book."""
        empty_ob = OrderBook(symbol="ERR/USDT", exchange="binance", bids=[], asks=[], timestamp=datetime.now())
        report = calculate_market_impact(empty_ob, side="buy", size=1.0)
        self.assertEqual(report.warning, "Order book empty")

    def test_calculate_market_impact_limits_violation(self):
        """Test exchange limit warnings."""
        limits = {"amount": {"min": 10.0}}
        report = calculate_market_impact(self.orderbook, side="buy", size=1.0, market_limits=limits)
        self.assertIn("below exchange minimum", report.warning)

    # ========== Synthetic Book Tests ==========

    def test_reconstruct_synthetic_book(self):
        """Test OHLCV to Synthetic book reconstruction."""
        # [timestamp, open, high, low, close, volume]
        candle = [1700000000000, 100, 110, 90, 105, 5000]
        synthetic = reconstruct_synthetic_book(candle, self.symbol)
        
        self.assertEqual(synthetic.mid_price, 105)
        # Range = 20. Spread bps = (20/105) * 10000 = 1904.76
        self.assertAlmostEqual(synthetic.estimated_spread_bps, 1904.76, places=1)

    # ========== Backtest Simulation Tests ==========

    def test_simulate_historical_execution(self):
        """Test historical execution aggregation."""
        candles = [
            [1700000000000, 100, 101, 99, 100, 1000],
            [1700000060000, 100, 102, 98, 101, 2000]
        ]
        report = simulate_historical_execution(candles, self.symbol, order_size=1.0, side="buy")
        
        self.assertEqual(report.total_candles, 2)
        self.assertTrue(report.avg_slippage_bps >= 0)
        self.assertEqual(report.side, "buy")

    # ========== Cross-Venue Analysis Tests ==========

    async def test_compare_liquidity_across_venues_arbitrage(self):
        """Test arbitrage detection across venues."""
        # Mock 2 clients
        def setup_client(cid, ob, fee=0.1):
            c = AsyncMock()
            c.exchange = MagicMock()
            c.exchange.fees = {'trading': {'taker': fee / 100.0}}
            c.exchange.market.return_value = {"limits": {"amount": {"min": 0.01}, "cost": {"min": 0.1}}}
            c.exchange.price_to_precision.side_effect = lambda s, p: str(p)
            c.exchange.amount_to_precision.side_effect = lambda s, a: str(a)
            c.exchange_id = cid
            c.circuit_breaker.state = "CLOSED"
            c.fetch_order_book.return_value = ob
            return c

        client1 = setup_client("binance", self.orderbook, fee=0.1)
        
        # Client 2 has much higher prices
        ob2 = OrderBook(
            symbol=self.symbol,
            exchange="coinbase",
            bids=[OrderBookLevel(price=109.0, amount=10.0)],
            asks=[OrderBookLevel(price=110.0, amount=10.0)],
            timestamp=datetime.now(),
            taker_fee_pct=0.1
        )
        
        client2 = setup_client("coinbase", ob2)
        
        # Compare
        result = await compare_liquidity_across_venues(
            symbol=self.symbol,
            order_size=1.0,
            side="buy",
            exchange_clients=[client1, client2]
        )
        
        print(f"DEBUG: venues={len(result.venue_analyses)} eligible={len([v for v in result.venue_analyses if v.is_eligible])}")
        for v in result.venue_analyses:
            print(f"DEBUG: {v.exchange_id} fill={v.fill_price} fee={v.taker_fee_pct} ok={v.is_eligible} warnings={v.execution_warnings}")
        print(f"DEBUG: summary={result.comparison_summary}")

        self.assertEqual(result.recommended_venue, "binance")
        self.assertTrue(result.arbitrage_opportunity)
        self.assertTrue(result.potential_profit_pct > 0)

    async def test_compare_liquidity_across_venues_circuit_open(self):
        """Test venue exclusion when circuit is OPEN."""
        client1 = AsyncMock()
        client1.exchange_id = "binance"
        client1.circuit_breaker.state = "OPEN"
        
        result = await compare_liquidity_across_venues(
            symbol=self.symbol,
            order_size=1.0,
            side="buy",
            exchange_clients=[client1]
        )
        self.assertEqual(result.recommended_venue, "None")
        self.assertEqual(result.venue_analyses[0].circuit_state, "OPEN")
        self.assertFalse(result.venue_analyses[0].is_eligible)

    def test_calculate_market_impact_high_slippage_alert(self):
        """Test slippage threshold alert mapping."""
        # Buy 10.0 BTC to hit depth and high slippage
        report = calculate_market_impact(self.orderbook, side="buy", size=10.0, slippage_threshold_bps=10.0)
        self.assertIn("SLIPPAGE", report.warning)
        self.assertIn("exceeds available book depth", report.warning)

    def test_simulate_historical_execution_precision(self):
        """Test simulation with exchange precision."""
        mock_exchange = MagicMock()
        mock_exchange.price_to_precision.return_value = "100.55"
        mock_exchange.id = "binance"
        
        candles = [[1700000000000, 100, 101, 99, 100, 1000]]
        report = simulate_historical_execution(candles, self.symbol, order_size=1.0, side="buy", exchange=mock_exchange)
        self.assertEqual(report.avg_fill_price, 100.55)

    def test_simulate_historical_execution_missing_data(self):
        """Test error when no candles provided."""
        with self.assertRaises(ValueError):
            simulate_historical_execution([], self.symbol, 1.0, "buy")

    def test_simulate_historical_execution_limit_violation(self):
        """Test simulation triggers warnings for limit violations."""
        limits = {"amount": {"min": 5.0}}
        candles = [[1700000000000, 100, 101, 99, 100, 1000]]
        # Order size 1.0 < min 5.0
        report = simulate_historical_execution(candles, self.symbol, order_size=1.0, side="buy", market_limits=limits)
        self.assertTrue(any("below minimum" in w for w in report.execution_warnings))

    async def test_compare_liquidity_across_venues_no_eligible(self):
        """Test summary when no venues are eligible."""
        client1 = AsyncMock()
        client1.exchange_id = "failed_ex"
        client1.circuit_breaker.state = "CLOSED"
        client1.fetch_order_book.side_effect = Exception("API Error")
        
        result = await compare_liquidity_across_venues(
            symbol=self.symbol,
            order_size=1.0,
            side="buy",
            exchange_clients=[client1]
        )
        self.assertEqual(result.recommended_venue, "None")
        self.assertIn("No eligible venues found", result.comparison_summary)

if __name__ == '__main__':
    unittest.main()
