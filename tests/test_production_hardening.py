import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic_ai import ModelRetry
from agents.market_agent import SYSTEM_PROMPT
from data_engine.models import LiquidityScorecard, OrderBook, OrderBookLevel
from datetime import datetime, UTC

@pytest.mark.asyncio
async def test_production_hardening_prompt():
    """Verify that SYSTEM_PROMPT contains production hardening instructions."""
    assert "latency_ms" in SYSTEM_PROMPT
    assert "circuit_state" in SYSTEM_PROMPT
    assert "200 bps" in SYSTEM_PROMPT
    assert "confidence_score" in SYSTEM_PROMPT
    assert "system_health_status" in SYSTEM_PROMPT
    assert "precision" in SYSTEM_PROMPT

@pytest.mark.asyncio
async def test_scorecard_model_validation():
    """Verify that LiquidityScorecard requires new hardening fields."""
    # This should fail if validation is working
    with pytest.raises(Exception):
        LiquidityScorecard(
            symbol="BTC/USDT",
            exchange="binance",
            liquidity_score=8,
            spread_analysis="Tight",
            depth_analysis="Deep",
            estimated_slippage_percent=0.1,
            recommended_max_size="$100k",
            summary_analysis="Excellent"
        )
    
    # Valid scorecard with all Phase 4 fields
    scorecard = LiquidityScorecard(
        symbol="BTC/USDT",
        exchange="binance",
        liquidity_score=8,
        spread_analysis="Tight",
        depth_analysis="Deep",
        estimated_slippage_percent=0.1,
        recommended_max_size="$100k",
        summary_analysis="Excellent",
        confidence_score=0.9,
        system_health_status="HEALTHY"
    )
    assert scorecard.confidence_score == 0.9
    assert scorecard.system_health_status == "HEALTHY"

@pytest.mark.asyncio
async def test_orderbook_telemetry_fields():
    """Verify that OrderBook model tracks all Phase 4 telemetry."""
    ob = OrderBook(
        symbol="BTC/USDT",
        exchange="binance",
        timestamp=datetime.now(UTC),
        bids=[OrderBookLevel(price=90000, amount=1)],
        asks=[OrderBookLevel(price=90100, amount=1)],
        latency_ms=120.5,
        circuit_state="CLOSED",
        taker_fee_pct=0.075
    )
    assert ob.latency_ms == 120.5
    assert ob.circuit_state == "CLOSED"
    assert ob.taker_fee_pct == 0.075

@pytest.mark.asyncio
async def test_tool_docstring_telemetry():
    """Ensure consolidated tools have telemetry metadata in docstrings."""
    from agents.tools import get_order_book_depth, calculate_market_impact
    
    assert "TELEMETRY AWARENESS" in get_order_book_depth.__doc__
    assert "latency_ms" in get_order_book_depth.__doc__
    assert "CRITICAL" in calculate_market_impact.__doc__
    assert "precision" in calculate_market_impact.__doc__

@pytest.mark.asyncio
async def test_model_retry_on_open_circuit():
    """Verify tools raise ModelRetry when circuit is OPEN."""
    from agents.tools import get_order_book_depth
    
    mock_client = MagicMock()
    mock_client.circuit_breaker.state = "OPEN"
    
    with patch('data_engine.exchange_manager.get_client', AsyncMock(return_value=mock_client)):
        with pytest.raises(ModelRetry) as excinfo:
            await get_order_book_depth(None, symbol="BTC/USDT", exchange="binance")
        
        assert "currently suspended" in str(excinfo.value)
        assert "Circuit OPEN" in str(excinfo.value)

if __name__ == "__main__":
    # For quick CLI verification
    import sys
    pytest.main([__file__, "-v"])
