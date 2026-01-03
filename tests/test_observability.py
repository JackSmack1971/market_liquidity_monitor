import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
import logfire
import ccxt

# Ensure package is in path
import sys
import os
sys.path.append(os.getcwd())

from data_engine import ExchangeClient, OrderBook
from data_engine.models import OrderBookLevel
from data_engine.analytics import calculate_market_impact
from data_engine.circuit_breaker import CircuitBreaker
from config import settings

@pytest.mark.asyncio
async def test_slippage_metric_recording(capfire):
    """Test Case 1: Verify market_slippage_bps gauge is recorded."""
    # Temporarily enable logfire in settings
    with patch.object(settings, 'logfire_token', 'test-token'):
        orderbook = OrderBook(
            symbol="SOL/USDT",
            exchange="binance",
            bids=[OrderBookLevel(price=100.0, amount=10.0)],
            asks=[OrderBookLevel(price=101.0, amount=10.0)],
        )
        
        # Trigger calculation
        # Side=buy, Size=5. Filled at 101.0. Ref price 101.0. 
        # Actually impact is 0 if filled at best ask.
        # Let's make it walk the book.
        orderbook.asks = [
            OrderBookLevel(price=101.0, amount=2.0),
            OrderBookLevel(price=102.0, amount=3.0)
        ]
        # VWAP = (2*101 + 3*102) / 5 = (202 + 306) / 5 = 508 / 5 = 101.6
        # Ref = 101.0. Impact = 0.6 / 101.0 ~ 0.59% (~59 bps)
        
        calculate_market_impact(orderbook, side='buy', size=5.0)
        
        # Verify metric
        metrics = capfire.metrics_reader.get_metrics_data()
        found = False
        all_metric_names = []
        if metrics:
            for rm in metrics.resource_metrics:
                for sm in rm.scope_metrics:
                    for m in sm.metrics:
                        all_metric_names.append(m.name)
                        if m.name == "market_slippage_bps":
                            for dp in m.data.data_points:
                                if dp.attributes.get("symbol") == "SOL/USDT":
                                    assert dp.value > 0
                                    found = True
        if not found:
            print(f"DEBUG: All metrics found: {all_metric_names}")
        assert found, "market_slippage_bps metric not found"

@pytest.mark.asyncio
async def test_circuit_breaker_events_and_metrics(capfire):
    """Test Case 2: Verify circuit breaker trips and metrics."""
    with patch.object(settings, 'logfire_token', 'test-token'):
        cb = CircuitBreaker(name="test-exchange", failure_threshold=2)
        
        # Simulate 2 failures
        for _ in range(2):
            try:
                await cb.call(AsyncMock(side_effect=Exception("API Error")))
            except:
                pass
        
        assert cb.state == "OPEN"
        
        # Verify metrics
        metrics = capfire.metrics_reader.get_metrics_data()
        found = False
        all_metric_names = []
        if metrics:
            for rm in metrics.resource_metrics:
                for sm in rm.scope_metrics:
                    for m in sm.metrics:
                        all_metric_names.append(m.name)
                        if m.name == "circuit_breaker_trips_total":
                            for dp in m.data.data_points:
                                if dp.attributes.get("circuit") == "test-exchange":
                                    assert dp.value >= 1
                                    found = True
        if not found:
            print(f"DEBUG: All metrics found: {all_metric_names}")
        assert found, "circuit_breaker_trips_total metric not found"
        
        # Verify span
        spans = capfire.exporter.exported_spans_as_dict()
        opened_spans = [s for s in spans if s["name"] == "circuit_opened"]
        if not opened_spans:
            print(f"DEBUG: All span names: {[s['name'] for s in spans]}")
        assert len(opened_spans) > 0

@pytest.mark.asyncio
async def test_data_engine_manual_spans(capfire):
    """Test Case 3: Verify fetch_order_book spans with template attributes."""
    with patch.object(settings, 'logfire_token', 'test-token'):
        client = ExchangeClient(exchange_id="binance")
        
        mock_orderbook = {
            "bids": [[100.0, 1.5]],
            "asks": [[101.0, 1.0]],
        }
        
        with patch.object(client.exchange, 'fetch_order_book', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_orderbook
            # Ensure markets loaded doesn't stall
            with patch.object(client, 'ensure_markets_loaded', new_callable=AsyncMock):
                await client.fetch_order_book("BTC/USDT", limit=1)
            
        spans = capfire.exporter.exported_spans_as_dict()
        fetch_spans = [s for s in spans if s["name"] == "fetch_order_book:{symbol}@{exchange}"]
        if not fetch_spans:
            print(f"DEBUG: All span names: {[s['name'] for s in spans]}")
        assert len(fetch_spans) > 0
        assert fetch_spans[0]["attributes"]["symbol"] == "BTC/USDT"
        assert fetch_spans[0]["attributes"]["exchange"] == "binance"
        
        await client.close()

@pytest.mark.asyncio
async def test_exception_fidelity_capture(capfire):
    """Test Case 4: Verify exception recording on spans."""
    with patch.object(settings, 'logfire_token', 'test-token'):
        client = ExchangeClient(exchange_id="binance")
        
        with patch.object(client.exchange, 'fetch_order_book', side_effect=ccxt.NetworkError("Timeout")):
            with patch.object(client, 'ensure_markets_loaded', new_callable=AsyncMock):
                try:
                    await client.fetch_order_book("BTC/USDT")
                except ccxt.NetworkError:
                    pass
                
        spans = capfire.exporter.exported_spans_as_dict()
        # Find any span that resulted from the call
        has_exception = False
        for span in spans:
            events = span.get("events", [])
            if any(e["name"] == "exception" for e in events):
                has_exception = True
                break
        
        assert has_exception, f"No exception captured in spans: {[s['name'] for s in spans]}"
        await client.close()

def test_security_scrubbing_mock(capfire):
    """Test Case 5: Verify redaction of sensitive patterns."""
    # We rely on the global configuration for scrubbing if possible, 
    # but since this is a unit test, we can manually trigger a scrub-check
    # logfire testing fixture automatically redacts according to global config
    
    # Intentionally log something that SHOULD be redacted by our global config 
    # (EXCHANGE_API_SECRET pattern r'EXCHANGE_API_KEY')
    logfire.info("Credential is {EXCHANGE_API_KEY}", EXCHANGE_API_KEY="secret-123")
    
    spans = capfire.exporter.exported_spans_as_dict()
    found_span = None
    for s in spans:
        if "EXCHANGE_API_KEY" in s["name"] or "EXCHANGE_API_KEY" in s["attributes"]:
            found_span = s
            break
            
    assert found_span is not None, f"Scrubbing span not found. Spans: {[s['name'] for s in spans]}"
    
    attr_val = str(found_span["attributes"].get("EXCHANGE_API_KEY"))
    msg = found_span["name"]
    
    # Redaction usually replaces the value
    assert "[REDACTED]" in attr_val or "[REDACTED]" in msg or "secret-123" not in attr_val
