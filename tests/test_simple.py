import pytest
from market_liquidity_monitor.data_engine.models import OrderBook, OrderBookLevel
from logfire.testing import CaptureLogfire
import logfire

def test_simple_pydantic_creation():
    level = OrderBookLevel(price=100.0, amount=1.0)
    orderbook = OrderBook(
        symbol="BTC/USDT",
        exchange="binance",
        bids=[level],
        asks=[level]
    )
    assert orderbook.symbol == "BTC/USDT"

def test_simple_logfire_capture(capfire):
    logfire.info("hello world")
    spans = capfire.exporter.exported_spans_as_dict()
    assert len(spans) > 0
    assert spans[0]["name"] == "hello world"
