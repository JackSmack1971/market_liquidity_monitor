
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from market_liquidity_monitor.data_engine.database import DatabaseManager

@pytest.mark.asyncio
async def test_db_connection_fail_graceful():
    """Test that DB manager handles connection failure gracefully."""
    # Mock engine.begin() to raise an exception
    with patch('market_liquidity_monitor.data_engine.database.create_async_engine') as mock_create:
        mock_engine = MagicMock()
        mock_create.return_value = mock_engine
        # engine.begin() is an async context manager
        mock_engine.begin.side_effect = lambda: MockAsyncContextManager(None) # Won't be used since we raise
        # Actually, let's make it raise directly when used
        mock_engine.begin.side_effect = Exception("Connection refused")
        # Ensure dispose can be awaited
        mock_engine.dispose = AsyncMock()
        
        manager = DatabaseManager(url="postgresql+asyncpg://invalid:invalid@localhost:5432/invalid")
        await manager.connect()
        assert manager.is_active is False
        await manager.disconnect()

@pytest.mark.asyncio
async def test_db_manager_singleton_import():
    """Test that the global db_manager can be imported."""
    from market_liquidity_monitor.data_engine.database import db_manager
    assert db_manager is not None
