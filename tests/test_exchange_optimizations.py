import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import ccxt.async_support as ccxt
import asyncio
from data_engine.exchange import ExchangeClient, with_retry

@pytest.mark.asyncio
async def test_ensure_markets_loaded():
    client = ExchangeClient(exchange_id='binance')
    client.exchange.load_markets = AsyncMock()
    
    # First call should trigger loading
    await client.ensure_markets_loaded()
    assert client._markets_loaded is True
    client.exchange.load_markets.assert_called_once()
    
    # Second call should NOT trigger loading
    await client.ensure_markets_loaded()
    client.exchange.load_markets.assert_called_once()

@pytest.mark.asyncio
async def test_retry_on_rate_limit():
    client = ExchangeClient(exchange_id='binance')
    client.exchange.rateLimit = 100 # ms
    
    # Mock a function that fails once with RateLimitExceeded then succeeds
    mock_func = AsyncMock()
    # Updated to ensure we pass the 'self' argument correctly if needed, 
    # but here the decorator is used as a standalone or on a mock.
    mock_func.side_effect = [ccxt.RateLimitExceeded("Too many requests"), {"status": "ok"}]
    
    # Apply decorator manually for testing
    decorated = with_retry(retries=2)(mock_func)
    
    with patch('asyncio.sleep', AsyncMock()) as mock_sleep:
        # Note: the decorator expects 'self' as first arg because it's designed for methods
        result = await decorated(client) 
        assert result == {"status": "ok"}
        assert mock_func.call_count == 2
        mock_sleep.assert_called_with(10.0) # Should default to 10s if rateLimit is small

@pytest.mark.asyncio
async def test_retry_on_network_error():
    client = ExchangeClient(exchange_id='binance')
    
    mock_func = AsyncMock()
    mock_func.side_effect = [ccxt.NetworkError("Timeout"), {"status": "ok"}]
    
    decorated = with_retry(retries=2, backoff=0.1)(mock_func)
    
    with patch('asyncio.sleep', AsyncMock()) as mock_sleep:
        result = await decorated(client)
        assert result == {"status": "ok"}
        assert mock_func.call_count == 2
        mock_sleep.assert_called_with(0.1)

def test_validate_order_limits():
    client = ExchangeClient(exchange_id='binance')
    client._markets_loaded = True
    
    # Mock market limits
    client.exchange.market = MagicMock(return_value={
        'limits': {
            'amount': {'min': 0.1, 'max': 100.0},
            'cost': {'min': 10.0}
        }
    })
    
    # Valid order
    assert client.validate_order_limits('BTC/USDT', 1.0, 50000) is True
    
    # Invalid amount (too low)
    assert client.validate_order_limits('BTC/USDT', 0.05, 50000) is False
    
    # Invalid cost (too low)
    assert client.validate_order_limits('BTC/USDT', 0.2, 10) is False

@pytest.mark.asyncio
async def test_fetch_order_book_triggers_market_loading():
    client = ExchangeClient(exchange_id='binance')
    client.exchange.load_markets = AsyncMock()
    client.exchange.fetch_order_book = AsyncMock(return_value={
        'bids': [[100, 1]],
        'asks': [[101, 1]]
    })
    
    await client.fetch_order_book('BTC/USDT')
    
    assert client._markets_loaded is True
    client.exchange.load_markets.assert_called_once()
    client.exchange.fetch_order_book.assert_called_once()
