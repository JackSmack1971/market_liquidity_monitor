"""
PostgreSQL database management for Market Liquidity Monitor.
"""

import os
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Float, DateTime, select, desc
from datetime import datetime, UTC

# Database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+asyncpg://mlm_user:mlm_password@localhost:5432/market_liquidity"
)

# SQLAlchemy models
class Base(DeclarativeBase):
    pass

class HistoricalSnapshotModel(Base):
    """SQLAlchemy model for historical liquidity snapshots."""
    __tablename__ = "historical_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    exchange: Mapped[str] = mapped_column(String(50), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), index=True)

    best_bid: Mapped[float] = mapped_column(Float)
    best_ask: Mapped[float] = mapped_column(Float)
    spread: Mapped[float] = mapped_column(Float)
    spread_percentage: Mapped[float] = mapped_column(Float)
    mid_price: Mapped[float] = mapped_column(Float)

    bid_volume_10: Mapped[float] = mapped_column(Float)
    ask_volume_10: Mapped[float] = mapped_column(Float)
    total_volume_20: Mapped[float] = mapped_column(Float)

    liquidity_1pct_usd: Mapped[float] = mapped_column(Float)
    liquidity_2pct_usd: Mapped[float] = mapped_column(Float)
    imbalance_ratio: Mapped[float] = mapped_column(Float)

class DatabaseManager:
    """Manages asynchronous PostgreSQL connections and operations."""

    def __init__(self, url: str = DATABASE_URL):
        self.url = url
        self.engine = create_async_engine(url, echo=False)
        self.session_factory = async_sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )
        self._is_active = False

    async def connect(self):
        """Test connection and create tables."""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self._is_active = True
            print("✅ Connected to PostgreSQL database")
        except Exception as e:
            print(f"⚠️ PostgreSQL connection failed: {e}")
            self._is_active = False

    async def disconnect(self):
        """Close database engine."""
        await self.engine.dispose()
        self._is_active = False

    @property
    def is_active(self) -> bool:
        return self._is_active

    async def store_snapshot(self, snapshot_data: Dict[str, Any]):
        """Store a new snapshot in the database."""
        if not self._is_active:
            return

        async with self.session_factory() as session:
            async with session.begin():
                db_snapshot = HistoricalSnapshotModel(**snapshot_data)
                session.add(db_snapshot)

    async def get_snapshots(
        self, 
        symbol: str, 
        exchange: str, 
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Retrieve historical snapshots from the database."""
        if not self._is_active:
            return []

        cutoff = datetime.now(UTC) - timedelta(hours=hours)

        async with self.session_factory() as session:
            stmt = select(HistoricalSnapshotModel).where(
                HistoricalSnapshotModel.symbol == symbol,
                HistoricalSnapshotModel.exchange == exchange,
                HistoricalSnapshotModel.timestamp >= cutoff
            ).order_by(desc(HistoricalSnapshotModel.timestamp))

            result = await session.execute(stmt)
            snapshots = result.scalars().all()
            
            return [
                {
                    column.name: getattr(s, column.name) 
                    for column in HistoricalSnapshotModel.__table__.columns
                } 
                for s in snapshots
            ]

# Global database manager instance
db_manager = DatabaseManager()
