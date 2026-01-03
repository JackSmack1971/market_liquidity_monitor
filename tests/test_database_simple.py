
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from market_liquidity_monitor.data_engine.database import DatabaseManager

@pytest.mark.asyncio
async def test_db_connection_fail_graceful():
    """Test that DB manager handles connection failure gracefully."""
    # Use invalid URL
    invalid_manager = DatabaseManager(url="postgresql+asyncpg://invalid:invalid@localhost:5432/invalid")
    await invalid_manager.connect()
    assert invalid_manager.is_active is False
    await invalid_manager.disconnect()

@pytest.mark.asyncio
@patch('market_liquidity_monitor.data_engine.database.async_sessionmaker')
@patch('market_liquidity_monitor.data_engine.database.create_async_engine')
async def test_store_snapshot_mock(mock_create_engine, mock_sessionmaker):
    """Test storing snapshot with mocked DB."""
    manager = DatabaseManager()
    manager._is_active = True
    
    mock_session = AsyncMock()
    mock_sessionmaker.return_value = MagicMock(return_value=mock_session)
    
    snapshot_data = {"symbol": "BTC/USDT", "exchange": "binance", "best_bid": 100.0}
    await manager.store_snapshot(snapshot_data)
    
    assert mock_session.add.called

@pytest.mark.asyncio
async def test_db_manager_singleton_import():
    """Test that the global db_manager can be imported."""
    from market_liquidity_monitor.data_engine.database import db_manager
    assert db_manager is not None
