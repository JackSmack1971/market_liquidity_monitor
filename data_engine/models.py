"""
Data models for market data.

Uses Pydantic for type-safe data structures.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class OrderBookLevel(BaseModel):
    """Single level in order book (bid or ask)."""

    price: float = Field(..., description="Price level")
    amount: float = Field(..., description="Volume at this price level")

    @property
    def total_value(self) -> float:
        """Calculate total value (price * amount)."""
        return self.price * self.amount


class OrderBook(BaseModel):
    """Order book snapshot."""

    symbol: str = Field(..., description="Trading pair symbol (e.g., SOL/USDT)")
    exchange: str = Field(..., description="Exchange name")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    bids: List[OrderBookLevel] = Field(..., description="Buy orders (descending price)")
    asks: List[OrderBookLevel] = Field(..., description="Sell orders (ascending price)")

    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        """Get best bid (highest buy price)."""
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        """Get best ask (lowest sell price)."""
        return self.asks[0] if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None

    @property
    def spread_percentage(self) -> Optional[float]:
        """Calculate spread as percentage of mid price."""
        if self.best_bid and self.best_ask and self.spread:
            mid_price = (self.best_bid.price + self.best_ask.price) / 2
            return (self.spread / mid_price) * 100
        return None

    def get_depth(self, side: str, levels: int = 10) -> List[OrderBookLevel]:
        """
        Get order book depth for a side.

        Args:
            side: 'bids' or 'asks'
            levels: Number of levels to return

        Returns:
            List of order book levels
        """
        data = self.bids if side == "bids" else self.asks
        return data[:levels]

    def get_cumulative_volume(self, side: str, levels: int = 10) -> float:
        """
        Calculate cumulative volume for a side.

        Args:
            side: 'bids' or 'asks'
            levels: Number of levels to include

        Returns:
            Total volume across levels
        """
        depth = self.get_depth(side, levels)
        return sum(level.amount for level in depth)

    def get_liquidity_at_percentage(self, side: str, percentage: float) -> tuple[float, float]:
        """
        Calculate available liquidity within percentage from best price.

        Args:
            side: 'bids' or 'asks'
            percentage: Percentage distance from best price (e.g., 1.0 for 1%)

        Returns:
            Tuple of (total_volume, total_value) within the percentage range
        """
        best_price = self.best_bid.price if side == "bids" else self.best_ask.price
        if not best_price:
            return 0.0, 0.0

        threshold = best_price * (1 + percentage / 100) if side == "asks" else best_price * (1 - percentage / 100)

        total_volume = 0.0
        total_value = 0.0

        levels = self.bids if side == "bids" else self.asks
        for level in levels:
            if (side == "bids" and level.price >= threshold) or (side == "asks" and level.price <= threshold):
                total_volume += level.amount
                total_value += level.total_value
            else:
                break

        return total_volume, total_value


class LiquidityAnalysis(BaseModel):
    """Analysis of order book liquidity."""

    symbol: str
    exchange: str
    timestamp: datetime

    # Spread metrics
    spread: float = Field(..., description="Absolute spread")
    spread_percentage: float = Field(..., description="Spread as % of mid price")

    # Depth metrics
    bid_depth_10: float = Field(..., description="Total volume in top 10 bids")
    ask_depth_10: float = Field(..., description="Total volume in top 10 asks")

    # Liquidity at different price ranges
    liquidity_1pct: tuple[float, float] = Field(
        ..., description="(volume, value) within 1% of best price"
    )
    liquidity_2pct: tuple[float, float] = Field(
        ..., description="(volume, value) within 2% of best price"
    )

    # Market impact estimation
    estimated_slippage_1k: Optional[float] = Field(
        None, description="Estimated slippage for $1k order"
    )
    estimated_slippage_10k: Optional[float] = Field(
        None, description="Estimated slippage for $10k order"
    )

    # Qualitative assessment
    liquidity_score: str = Field(
        ..., description="HIGH, MEDIUM, LOW based on metrics"
    )
    reasoning: str = Field(..., description="Human-readable analysis from LLM")


class MarketQuery(BaseModel):
    """User query about market liquidity."""

    query: str = Field(..., description="Natural language query")
    symbol: Optional[str] = Field(None, description="Specific trading pair if mentioned")
    exchange: Optional[str] = Field(None, description="Specific exchange if mentioned")


class ExchangeComparison(BaseModel):
    """Comparison of liquidity across multiple exchanges."""

    symbol: str = Field(..., description="Trading pair symbol")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Per-exchange data
    exchanges: List[str] = Field(..., description="List of exchanges compared")
    order_books: List[OrderBook] = Field(..., description="Order books from each exchange")

    # Comparative metrics
    best_bid_exchange: str = Field(..., description="Exchange with highest bid")
    best_ask_exchange: str = Field(..., description="Exchange with lowest ask")
    tightest_spread_exchange: str = Field(..., description="Exchange with tightest spread")
    deepest_liquidity_exchange: str = Field(..., description="Exchange with most depth")

    # Arbitrage opportunity
    arbitrage_opportunity: Optional[float] = Field(
        None, description="Potential arbitrage profit percentage"
    )
    arbitrage_route: Optional[str] = Field(
        None, description="Buy on X, sell on Y route"
    )

    # Aggregated analysis
    average_spread_pct: float = Field(..., description="Average spread across exchanges")
    total_liquidity_usd: float = Field(..., description="Combined liquidity in USD")

    # LLM synthesis
    recommendation: str = Field(
        ..., description="Which exchange to use and why"
    )
    reasoning: str = Field(..., description="Detailed comparative analysis")


class HistoricalSnapshot(BaseModel):
    """Historical liquidity snapshot for time-series tracking."""

    symbol: str
    exchange: str
    timestamp: datetime

    # Price metrics
    best_bid: float
    best_ask: float
    spread: float
    spread_percentage: float
    mid_price: float

    # Volume metrics
    bid_volume_10: float = Field(..., description="Volume in top 10 bids")
    ask_volume_10: float = Field(..., description="Volume in top 10 asks")
    total_volume_20: float = Field(..., description="Combined volume in 20 levels")

    # Liquidity depth
    liquidity_1pct_usd: float = Field(..., description="USD liquidity within 1% of mid")
    liquidity_2pct_usd: float = Field(..., description="USD liquidity within 2% of mid")

    # Market health indicators
    imbalance_ratio: float = Field(
        ..., description="Bid volume / Ask volume ratio (>1 = buying pressure)"
    )

    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LiquidityAlert(BaseModel):
    """Alert triggered by liquidity anomaly."""

    alert_id: str = Field(..., description="Unique alert identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    severity: str = Field(..., description="HIGH, MEDIUM, LOW")

    symbol: str
    exchange: str

    # Alert type
    alert_type: str = Field(
        ...,
        description="SPREAD_WIDENING, DEPTH_DROP, IMBALANCE, PRICE_ANOMALY"
    )

    # Metrics
    current_value: float = Field(..., description="Current metric value")
    baseline_value: float = Field(..., description="Historical baseline")
    deviation_percentage: float = Field(..., description="% deviation from baseline")

    # Context
    message: str = Field(..., description="Human-readable alert message")
    suggested_action: Optional[str] = Field(
        None, description="Recommended action for trader"
    )

    # Metadata
    requires_action: bool = Field(default=False)
    acknowledged: bool = Field(default=False)
