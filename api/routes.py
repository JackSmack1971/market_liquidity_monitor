"""
API routes for market liquidity monitoring.

Provides endpoints for:
- Order book data
- Liquidity analysis
- Natural language queries
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional

from ..data_engine import ExchangeClient, OrderBook
from ..data_engine.models import LiquidityAnalysis, MarketQuery
from ..agents import MarketAnalyzer
from .dependencies import get_exchange_client, get_market_analyzer

router = APIRouter()


@router.get("/orderbook/{symbol}", response_model=OrderBook)
async def get_orderbook(
    symbol: str,
    exchange: str = Query(default="binance", description="Exchange name"),
    levels: int = Query(default=20, ge=1, le=100, description="Number of levels"),
    client: ExchangeClient = Depends(get_exchange_client),
) -> OrderBook:
    """
    Fetch real-time order book for a trading pair.

    Args:
        symbol: Trading pair (e.g., 'SOL/USDT')
        exchange: Exchange name
        levels: Number of order book levels

    Returns:
        Order book with bids, asks, and computed metrics

    Example:
        GET /orderbook/SOL/USDT?exchange=binance&levels=20
    """
    try:
        orderbook = await client.fetch_order_book(symbol, limit=levels)
        return orderbook
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch order book: {str(e)}"
        )


@router.get("/search/{query}", response_model=list[str])
async def search_symbols(
    query: str,
    exchange: str = Query(default="binance", description="Exchange name"),
    client: ExchangeClient = Depends(get_exchange_client),
) -> list[str]:
    """
    Search for trading pairs matching a query.

    Args:
        query: Search term (e.g., 'SOL', 'BTC')
        exchange: Exchange name

    Returns:
        List of matching trading pair symbols

    Example:
        GET /search/SOL?exchange=binance
    """
    try:
        symbols = await client.search_symbol(query)
        return symbols
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search symbols: {str(e)}"
        )


@router.post("/analyze", response_model=LiquidityAnalysis)
async def analyze_liquidity(
    query: MarketQuery,
    analyzer: MarketAnalyzer = Depends(get_market_analyzer),
) -> LiquidityAnalysis:
    """
    Analyze market liquidity using natural language query.

    This endpoint uses LLM reasoning to:
    1. Parse the query and extract intent
    2. Fetch relevant market data
    3. Analyze liquidity metrics
    4. Provide human-readable insights

    Args:
        query: Market query with natural language description

    Returns:
        Liquidity analysis with metrics and reasoning

    Example:
        POST /analyze
        {
            "query": "How is the SOL liquidity on Binance?",
            "symbol": "SOL/USDT",
            "exchange": "binance"
        }
    """
    try:
        analysis = await analyzer.analyze_liquidity(
            query=query.query,
            symbol=query.symbol,
            exchange=query.exchange,
        )
        return analysis
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze liquidity: {str(e)}"
        )


@router.get("/quick-check/{symbol}", response_model=dict)
async def quick_liquidity_check(
    symbol: str,
    exchange: str = Query(default="binance", description="Exchange name"),
    analyzer: MarketAnalyzer = Depends(get_market_analyzer),
) -> dict:
    """
    Quick liquidity check for a trading pair.

    Args:
        symbol: Trading pair (e.g., 'SOL/USDT')
        exchange: Exchange name

    Returns:
        Brief assessment

    Example:
        GET /quick-check/SOL/USDT?exchange=binance
    """
    try:
        assessment = await analyzer.quick_check(symbol, exchange)
        return {"symbol": symbol, "exchange": exchange, "assessment": assessment}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check liquidity: {str(e)}"
        )


@router.post("/estimate-slippage", response_model=LiquidityAnalysis)
async def estimate_slippage(
    symbol: str,
    order_size_usd: float = Query(..., gt=0, description="Order size in USD"),
    side: str = Query(default="buy", regex="^(buy|sell)$"),
    exchange: str = Query(default="binance", description="Exchange name"),
    analyzer: MarketAnalyzer = Depends(get_market_analyzer),
) -> LiquidityAnalysis:
    """
    Estimate slippage for a potential order.

    Args:
        symbol: Trading pair
        order_size_usd: Order size in USD
        side: 'buy' or 'sell'
        exchange: Exchange name

    Returns:
        Analysis with slippage estimation

    Example:
        POST /estimate-slippage?symbol=SOL/USDT&order_size_usd=1000&side=buy
    """
    try:
        analysis = await analyzer.estimate_slippage(
            symbol=symbol,
            order_size_usd=order_size_usd,
            side=side,
            exchange=exchange,
        )
        return analysis
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to estimate slippage: {str(e)}"
        )


@router.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint.

    Returns:
        Status information
    """
    return {
        "status": "healthy",
        "service": "market-liquidity-monitor",
    }


@router.post("/compare-exchanges", response_model=dict)
async def compare_exchanges_endpoint(
    symbol: str = Query(..., description="Trading pair symbol"),
    exchanges: list[str] = Query(
        default=["binance", "coinbase", "kraken"],
        description="List of exchanges to compare"
    ),
    levels: int = Query(default=20, ge=1, le=100, description="Order book depth"),
) -> dict:
    """
    Compare liquidity across multiple exchanges simultaneously.

    Fetches order books from multiple exchanges in parallel and provides:
    - Best bid/ask across exchanges
    - Tightest spread
    - Deepest liquidity
    - Arbitrage opportunities
    - Comparative analysis

    Args:
        symbol: Trading pair (e.g., 'SOL/USDT')
        exchanges: List of exchange names
        levels: Number of order book levels

    Returns:
        Comprehensive comparison with recommendations

    Example:
        POST /compare-exchanges?symbol=SOL/USDT&exchanges=binance&exchanges=coinbase
    """
    try:
        from ..agents.tools import compare_exchanges

        comparison = await compare_exchanges(
            symbol=symbol,
            exchanges=exchanges,
            levels=levels
        )
        return comparison
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compare exchanges: {str(e)}"
        )


@router.post("/market-impact", response_model=dict)
async def calculate_market_impact_endpoint(
    symbol: str = Query(..., description="Trading pair symbol"),
    order_size_usd: float = Query(..., gt=0, description="Order size in USD"),
    side: str = Query(default="buy", regex="^(buy|sell)$"),
    exchange: str = Query(default="binance", description="Exchange name"),
) -> dict:
    """
    Calculate detailed market impact for a potential order.

    Simulates order execution through the order book to determine:
    - Average execution price
    - Slippage percentage
    - Price impact
    - Number of levels consumed
    - Liquidity sufficiency

    Args:
        symbol: Trading pair
        order_size_usd: Order size in USD
        side: 'buy' or 'sell'
        exchange: Exchange name

    Returns:
        Detailed market impact analysis

    Example:
        POST /market-impact?symbol=SOL/USDT&order_size_usd=10000&side=buy
    """
    try:
        from ..agents.tools import calculate_market_impact

        impact = await calculate_market_impact(
            symbol=symbol,
            order_size_usd=order_size_usd,
            side=side,
            exchange=exchange
        )
        return impact
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate market impact: {str(e)}"
        )
