"""
Data models for market data.

Uses Pydantic for type-safe data structures.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict
from datetime import datetime, timezone, UTC


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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

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


class LiquidityScorecard(BaseModel):
    """
    Structured scorecard for liquidity analysis.
    
    This model is used by the MarketAgent to provide consistent, machine-readable validation
    of market conditions.
    """
    
    symbol: str = Field(..., description="Trading pair analyzed")
    exchange: str = Field(..., description="Primary exchange analyzed")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Analysis timestamp")
    
    liquidity_score: int = Field(
        ..., 
        ge=1, 
        le=10, 
        description="Overall liquidity rating (1-10). 10 = Excellent depth/tight spread."
    )
    
    spread_analysis: str = Field(..., description="Analysis of bid-ask spread health")
    depth_analysis: str = Field(..., description="Analysis of order book depth and resilience")
    
    estimated_slippage_percent: float = Field(
        ..., 
        description="Expected slippage for a standard $10k market order"
    )
    
    recommended_max_size: str = Field(
        ..., 
        description="Recommended maximum order size to avoid significant impact (e.g., '$50,000')"
    )
    
    risk_factors: List[str] = Field(
        default_factory=list, 
        description="List of detected risks (e.g., 'Low Volume', 'Asymmetric Depth', 'Thin Orderbook')"
    )
    
    summary_analysis: str = Field(
        ..., 
        description="Executive summary of the market conditions and execution advice."
    )
    
    market_impact_report: Optional['MarketImpactReport'] = Field(
        None, 
        description="Detailed slippage analysis if a specific trade size was simulated."
    )

class MarketImpactReport(BaseModel):
    """
    Detailed report on market impact and slippage for a specific order size.
    """
    symbol: str = Field(..., description="Trading pair")
    side: str = Field(..., description="Order side (buy or sell)")
    target_size: float = Field(..., description="Requested order size in base currency")
    target_value_usd: float = Field(..., description="Requested order value in USD")
    
    expected_fill_price: float = Field(..., description="Volume Weighted Average Price (VWAP) of the fill")
    reference_price: float = Field(..., description="Mid price at start of simulation")
    
    slippage_bps: float = Field(..., description="Slippage in basis points (1 bp = 0.01%)")
    price_impact_percent: float = Field(..., description="Price impact as percentage")
    
    critical_depth_level: Optional[float] = Field(None, description="Price level reached to fill order")
    
    warning: Optional[str] = Field(None, description="Warning if order exceeds available book depth")

class VenueAnalysis(BaseModel):
    """
    Analysis of liquidity and execution quality for a single exchange venue.
    """
    exchange_id: str = Field(..., description="Exchange identifier (e.g., 'binance', 'kraken')")
    fill_price: float = Field(..., description="Expected fill price (VWAP) with exchange precision applied")
    slippage_bps: float = Field(..., description="Slippage in basis points")
    execution_warnings: List[str] = Field(default_factory=list, description="Execution risks or limit violations")
    circuit_state: str = Field(..., description="Circuit breaker state (CLOSED/OPEN/HALF_OPEN)")
    is_eligible: bool = Field(..., description="Whether venue can handle the order size")
    taker_fee_pct: float = Field(default=0.1, description="Taker fee percentage (e.g., 0.1 for 0.1%)")

class CrossExchangeComparison(BaseModel):
    """
    Comprehensive comparison of liquidity across multiple exchanges.
    """
    symbol: str = Field(..., description="Trading pair analyzed")
    order_size: float = Field(..., description="Simulated order size in base currency")
    side: str = Field(..., description="Order side (buy or sell)")
    
    recommended_venue: str = Field(..., description="Optimal exchange for execution based on lowest slippage")
    
    arbitrage_opportunity: bool = Field(default=False, description="Whether cross-exchange arbitrage is possible")
    potential_profit_pct: Optional[float] = Field(None, description="Estimated arbitrage profit percentage (fee-adjusted)")
    
    venue_analyses: List[VenueAnalysis] = Field(..., description="Detailed analysis for each venue")
    comparison_summary: str = Field(..., description="Executive summary of routing recommendation")


class SyntheticOrderBook(BaseModel):
    """
    Reconstructed order book from historical OHLCV data.
    """
    timestamp: datetime = Field(..., description="Candle timestamp")
    symbol: str = Field(..., description="Trading pair")
    mid_price: float = Field(..., description="Mid price (close of candle)")
    estimated_spread_bps: float = Field(..., description="Estimated spread in basis points based on volatility")
    volatility_percentile: float = Field(..., description="Volatility percentile (0-100)")
    synthetic_depth_levels: int = Field(default=10, description="Number of synthetic depth levels generated")
    high: float = Field(..., description="Candle high")
    low: float = Field(..., description="Candle low")
    volume: float = Field(..., description="Candle volume")

class BacktestReport(BaseModel):
    """
    Comprehensive report of historical execution simulation.
    """
    symbol: str = Field(..., description="Trading pair analyzed")
    exchange: str = Field(..., description="Exchange used for backtest")
    timeframe: str = Field(..., description="Candle timeframe (e.g., '1h', '4h', '1d')")
    start_date: datetime = Field(..., description="Backtest start timestamp")
    end_date: datetime = Field(..., description="Backtest end timestamp")
    
    order_size: float = Field(..., description="Simulated order size in base currency")
    side: str = Field(..., description="Order side (buy or sell)")
    
    total_candles: int = Field(..., description="Number of candles analyzed")
    avg_slippage_bps: float = Field(..., description="Average slippage across all candles")
    max_slippage_bps: float = Field(..., description="Maximum slippage encountered")
    min_slippage_bps: float = Field(..., description="Minimum slippage encountered")
    
    avg_fill_price: float = Field(..., description="Average fill price across backtest")
    theoretical_total_cost: float = Field(..., description="Total cost if all orders executed")
    
    execution_warnings: List[str] = Field(default_factory=list, description="Warnings from simulation")
    volatility_profile: Dict[str, float] = Field(default_factory=dict, description="Volatility statistics")
    
    high_risk_periods: int = Field(default=0, description="Number of candles with >200 bps slippage")
    optimal_execution_windows: int = Field(default=0, description="Number of candles with <50 bps slippage")




class HistoricalLiquidityTrend(BaseModel):
    """
    Analysis of historical liquidity trends over a specific period.
    """
    period_summary: str = Field(description="Narrative of liquidity behavior over the period")
    volatility_index: float = Field(ge=0, description="Standard deviation of price movements")
    volume_consistency_score: int = Field(ge=1, le=10, description="1-10 score of volume stability (10 = very stable)")
    significant_outliers: List[str] = Field(description="Specific timestamps of liquidity shocks or volume spikes")
    volatility_rating: str = Field(..., description="Market stability assessment (STABLE/MODERATE/VOLATILE)")


class LiquidityAnalysis(BaseModel):
    """Refined analysis of market liquidity with structured metrics for UI rendering."""

    symbol: str = Field(..., description="The trading pair symbol analyzed (e.g., SOL/USDT)")
    exchange: str = Field(..., description="The exchange where the data was sourced from")
    timestamp: datetime = Field(..., description="The exact time of this analysis")

    # Spread metrics
    spread: float = Field(..., description="The absolute difference between best bid and best ask")
    spread_percentage: float = Field(..., description="The bid-ask spread as a percentage of the mid price")

    # Depth metrics
    bid_depth_10: float = Field(..., description="Combined volume of the top 10 bid levels")
    ask_depth_10: float = Field(..., description="Combined volume of the top 10 ask levels")

    # Liquidity at different price ranges
    liquidity_1pct: tuple[float, float] = Field(
        ..., description="Available (volume, USD value) within 1% of the best price"
    )
    liquidity_2pct: tuple[float, float] = Field(
        ..., description="Available (volume, USD value) within 2% of the best price"
    )

    # Market impact estimation
    estimated_slippage_1k: Optional[float] = Field(
        None, description="Predicted slippage percentage for a $1,000 market order"
    )
    estimated_slippage_10k: Optional[float] = Field(
        None, description="Predicted slippage percentage for a $10,000 market order"
    )

    # Risk Metrics
    volatility_rating: str = Field(
        ..., description="Current market volatility assessment (STABLE, MODERATE, VOLATILE)"
    )

    # Qualitative assessment
    liquidity_score: str = Field(
        ..., description="Overall liquidity grade: HIGH (Easy to trade), MEDIUM (Use caution), LOW (High slippage risk)"
    )
    reasoning: str = Field(..., description="Detailed technical justification for the assigned liquidity score and metrics")


class MarketQuery(BaseModel):
    """User query about market liquidity."""

    query: str = Field(..., description="Natural language query")
    symbol: Optional[str] = Field(None, description="Specific trading pair if mentioned")
    exchange: Optional[str] = Field(None, description="Specific exchange if mentioned")


class ExchangeComparison(BaseModel):
    """Comparison of liquidity across multiple exchanges."""

    symbol: str = Field(..., description="Trading pair symbol")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

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

    model_config = ConfigDict(
        populate_by_name=True,
    )


class LiquidityAlert(BaseModel):
    """Alert triggered by liquidity anomaly."""

    alert_id: str = Field(..., description="Unique alert identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
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
