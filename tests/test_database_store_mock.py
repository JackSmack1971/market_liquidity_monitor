import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from market_liquidity_monitor.data_engine.database import DatabaseManager

class MockAsyncContextManager:
    def __init__(self, value):
        self.value = value
    async def __aenter__(self):
        return self.value
    async def __aexit__(self, *args, **kwargs):
        pass

class MockBegin:
    async def __aenter__(self):
        return None
    async def __aexit__(self, *args, **kwargs):
        pass

class MockSession:
    def __init__(self):
        self.add = MagicMock()
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args, **kwargs):
        pass
    def begin(self):
        return MockBegin()

@pytest.mark.asyncio
@patch('market_liquidity_monitor.data_engine.database.async_sessionmaker')
@patch('market_liquidity_monitor.data_engine.database.create_async_engine')
async def test_store_snapshot_mock(mock_create_engine, mock_sessionmaker):
    """Test storing snapshot with mocked DB."""
    mock_session = MockSession()
    # We MUST use MagicMock here so that calling factory_instance() is NOT async.
    factory_instance = MagicMock(return_value=MockAsyncContextManager(mock_session))
    mock_sessionmaker.return_value = factory_instance
    
    manager = DatabaseManager()
    manager._is_active = True
    
    snapshot_data = {"symbol": "BTC/USDT", "exchange": "binance", "best_bid": 100.0}
    await manager.store_snapshot(snapshot_data)
    
    assert mock_session.add.called
