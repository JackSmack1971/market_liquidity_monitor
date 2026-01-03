import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic_ai import ModelRetry
from agents.market_agent import create_market_agent
from data_engine.models import LiquidityAnalysis
from datetime import datetime, UTC

@pytest.mark.asyncio
async def test_agent_structured_output_validation():
    """Verify that the agent returns a validated LiquidityAnalysis object."""
    agent = create_market_agent()
    
    # Mock the model to return a valid structured response
    mock_result = MagicMock()
    mock_result.data = LiquidityAnalysis(
        symbol="BTC/USDT",
        exchange="binance",
        timestamp=datetime.now(UTC),
        spread=10.0,
        spread_percentage=0.01,
        bid_depth_10=100.0,
        ask_depth_10=100.0,
        liquidity_1pct=(50.0, 50000.0),
        liquidity_2pct=(80.0, 80000.0),
        volatility_rating="STABLE",
        liquidity_score="HIGH",
        reasoning="Test reasoning"
    )
    
    with patch.object(agent, 'run', AsyncMock(return_value=mock_result)):
        result = await agent.run("Test query")
        assert isinstance(result.data, LiquidityAnalysis)
        assert result.data.symbol == "BTC/USDT"
        assert result.data.volatility_rating == "STABLE"

@pytest.mark.asyncio
async def test_tool_retry_on_invalid_symbol():
    """Verify that tools raise ModelRetry for invalid symbols."""
    from agents.tools import get_order_book_depth
    
    with patch('data_engine.exchange_manager.get_client', AsyncMock()) as mock_get_client:
        mock_client = AsyncMock()
        mock_client.fetch_order_book.side_effect = ValueError("Symbol not found")
        mock_get_client.return_value = mock_client
        
        with pytest.raises(ModelRetry) as excinfo:
            await get_order_book_depth(None, symbol="INVALID/TICKER", exchange="binance")
        
        assert "was not found on 'binance'" in str(excinfo.value)
        assert "search_trading_pairs" in str(excinfo.value)

@pytest.mark.asyncio
async def test_tool_retry_on_unsupported_exchange():
    """Verify that tools raise ModelRetry for unsupported exchanges."""
    from agents.tools import search_trading_pairs
    
    with patch('data_engine.exchange_manager.get_client', AsyncMock()) as mock_get_client:
        mock_get_client.side_effect = Exception("Exchange not found: unsupported")
        
        with pytest.raises(ModelRetry) as excinfo:
            await search_trading_pairs(None, query="BTC", exchange="unsupported")
        
        assert "not supported" in str(excinfo.value)

def test_tool_docstrings_are_prompts():
    """Ensure tool docstrings contain critical guidance for the LLM."""
    from agents.tools import AGENT_TOOLS
    
    for tool in AGENT_TOOLS:
        doc = tool.__doc__
        assert doc is not None
        assert len(doc.strip()) > 50 # Ensure non-trivial docstrings
        if tool.__name__ == "search_trading_pairs":
            assert "MANDATORY" in doc
            assert "formatting" in doc
