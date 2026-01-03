"""
Tool definitions for the LLM agent.

These tools allow the agent to fetch real-time market data and perform analysis.
"""

from typing import Optional, List
from pydantic import Field
import asyncio

from ..data_engine import exchange_manager, OrderBook
from ..data_engine.models import ExchangeComparison


async def get_order_book_depth(
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'SOL/USDT', 'BTC/USDT')"),
    exchange: str = Field(default="binance", description="Exchange name (e.g., 'binance', 'coinbase')"),
    levels: int = Field(default=20, description="Number of order book levels to fetch (default: 20)"),
) -> OrderBook:
    """
    Fetch real-time order book depth for a trading pair.

    This tool retrieves the current bids (buy orders) and asks (sell orders)
    from the specified exchange, allowing analysis of market liquidity.

    Args:
        symbol: Trading pair (e.g., 'SOL/USDT')
        exchange: Exchange to query
        levels: Number of price levels to retrieve

    Returns:
        OrderBook object with bid/ask data and computed metrics
    """
    client = await exchange_manager.get_client(exchange)
    orderbook = await client.fetch_order_book(symbol, limit=levels)
    return orderbook


async def search_trading_pairs(
    query: str = Field(..., description="Search term (e.g., 'SOL', 'BTC', 'ETH')"),
    exchange: str = Field(default="binance", description="Exchange to search"),
) -> list[str]:
    """
    Search for available trading pairs on an exchange.

    Useful when the user mentions a token but doesn't specify the full pair
    (e.g., user says "SOL" and we need to find "SOL/USDT").

    Args:
        query: Token symbol or search term
        exchange: Exchange to search

    Returns:
        List of matching trading pair symbols
    """
    client = await exchange_manager.get_client(exchange)
    symbols = await client.search_symbol(query)
    return symbols


async def get_market_metadata(
    symbol: str = Field(..., description="Trading pair symbol"),
    exchange: str = Field(default="binance", description="Exchange name"),
) -> dict:
    """
    Get detailed market information for a trading pair.

    Provides metadata like precision requirements, trading limits,
    and fee structure.

    Args:
        symbol: Trading pair
        exchange: Exchange name

    Returns:
        Market metadata dictionary
    """
    client = await exchange_manager.get_client(exchange)
    info = await client.get_market_info(symbol)
    return {
        "symbol": info.get("symbol"),
        "base": info.get("base"),
        "quote": info.get("quote"),
        "active": info.get("active"),
        "precision": info.get("precision"),
        "limits": info.get("limits"),
    }


async def compare_exchanges(
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'SOL/USDT')"),
    exchanges: List[str] = Field(
        default=["binance", "coinbase", "kraken"],
        description="List of exchanges to compare (e.g., ['binance', 'coinbase', 'kraken'])"
    ),
    levels: int = Field(default=20, description="Number of order book levels to fetch"),
) -> dict:
    """
    Compare liquidity across multiple exchanges simultaneously.

    This tool fetches order books from multiple exchanges in parallel and
    performs comparative analysis to identify the best exchange for trading,
    arbitrage opportunities, and liquidity differences.

    Args:
        symbol: Trading pair to compare
        exchanges: List of exchange names to query
        levels: Number of order book levels

    Returns:
        Dictionary with comparison data including best exchange recommendations
        and arbitrage opportunities
    """
    # Fetch order books in parallel for performance
    async def fetch_from_exchange(exchange: str):
        try:
            client = await exchange_manager.get_client(exchange)
            orderbook = await client.fetch_order_book(symbol, limit=levels)
            return (exchange, orderbook, None)
        except Exception as e:
            return (exchange, None, str(e))

    results = await asyncio.gather(
        *[fetch_from_exchange(ex) for ex in exchanges],
        return_exceptions=True
    )

    # Separate successful and failed fetches
    successful_books = []
    failed_exchanges = []

    for exchange, orderbook, error in results:
        if orderbook:
            successful_books.append((exchange, orderbook))
        else:
            failed_exchanges.append({"exchange": exchange, "error": error})

    if not successful_books:
        return {
            "error": "Failed to fetch order books from all exchanges",
            "failed_exchanges": failed_exchanges
        }

    # Calculate comparative metrics
    exchange_names = [ex for ex, _ in successful_books]
    order_books = [ob for _, ob in successful_books]

    # Find best bid (highest)
    best_bid_data = max(
        [(ex, ob.best_bid.price if ob.best_bid else 0) for ex, ob in successful_books],
        key=lambda x: x[1]
    )
    best_bid_exchange = best_bid_data[0]

    # Find best ask (lowest)
    best_ask_data = min(
        [(ex, ob.best_ask.price if ob.best_ask else float('inf')) for ex, ob in successful_books],
        key=lambda x: x[1]
    )
    best_ask_exchange = best_ask_data[0]

    # Find tightest spread
    spreads = [
        (ex, ob.spread_percentage if ob.spread_percentage else float('inf'))
        for ex, ob in successful_books
    ]
    tightest_spread_data = min(spreads, key=lambda x: x[1])
    tightest_spread_exchange = tightest_spread_data[0]

    # Find deepest liquidity (combined volume in top 10 levels)
    depths = [
        (ex, ob.get_cumulative_volume("bids", 10) + ob.get_cumulative_volume("asks", 10))
        for ex, ob in successful_books
    ]
    deepest_data = max(depths, key=lambda x: x[1])
    deepest_liquidity_exchange = deepest_data[0]

    # Calculate arbitrage opportunity
    arbitrage_profit_pct = None
    arbitrage_route = None
    if best_bid_data[1] > best_ask_data[1]:
        # Arbitrage exists: buy on best_ask_exchange, sell on best_bid_exchange
        arbitrage_profit_pct = ((best_bid_data[1] - best_ask_data[1]) / best_ask_data[1]) * 100
        arbitrage_route = f"Buy on {best_ask_exchange} @ ${best_ask_data[1]:.2f}, Sell on {best_bid_exchange} @ ${best_bid_data[1]:.2f}"

    # Calculate average spread
    valid_spreads = [s for _, s in spreads if s != float('inf')]
    avg_spread = sum(valid_spreads) / len(valid_spreads) if valid_spreads else 0

    # Calculate total liquidity
    total_liquidity = sum(depth for _, depth in depths)

    return {
        "symbol": symbol,
        "exchanges_compared": exchange_names,
        "successful_fetches": len(successful_books),
        "failed_exchanges": failed_exchanges,
        "best_bid_exchange": best_bid_exchange,
        "best_bid_price": best_bid_data[1],
        "best_ask_exchange": best_ask_exchange,
        "best_ask_price": best_ask_data[1],
        "tightest_spread_exchange": tightest_spread_exchange,
        "tightest_spread_pct": tightest_spread_data[1],
        "deepest_liquidity_exchange": deepest_liquidity_exchange,
        "deepest_liquidity_volume": deepest_data[1],
        "arbitrage_opportunity_pct": arbitrage_profit_pct,
        "arbitrage_route": arbitrage_route,
        "average_spread_pct": avg_spread,
        "total_liquidity_volume": total_liquidity,
        "order_books": [
            {
                "exchange": ex,
                "spread_pct": ob.spread_percentage,
                "bid_depth": ob.get_cumulative_volume("bids", 10),
                "ask_depth": ob.get_cumulative_volume("asks", 10),
            }
            for ex, ob in successful_books
        ]
    }


async def calculate_market_impact(
    symbol: str = Field(..., description="Trading pair symbol"),
    order_size_usd: float = Field(..., description="Order size in USD"),
    side: str = Field(default="buy", description="'buy' or 'sell'"),
    exchange: str = Field(default="binance", description="Exchange name"),
) -> dict:
    """
    Calculate detailed market impact and slippage for a potential order.

    This tool simulates walking through the order book to determine:
    - Average execution price
    - Slippage percentage
    - Price impact on the market
    - Number of levels consumed

    Args:
        symbol: Trading pair
        order_size_usd: Order size in USD
        side: Trade direction ('buy' or 'sell')
        exchange: Exchange to analyze

    Returns:
        Dictionary with detailed market impact metrics
    """
    client = await exchange_manager.get_client(exchange)
    orderbook = await client.fetch_order_book(symbol, limit=100)

    # Determine which side of the book to use
    levels = orderbook.asks if side == "buy" else orderbook.bids
    best_price = orderbook.best_ask.price if side == "buy" else orderbook.best_bid.price

    if not levels or not best_price:
        return {"error": "No liquidity available"}

    # Simulate order execution
    remaining_usd = order_size_usd
    total_base_amount = 0.0
    total_usd_spent = 0.0
    levels_consumed = 0

    for level in levels:
        if remaining_usd <= 0:
            break

        level_value_usd = level.price * level.amount
        consumed_usd = min(remaining_usd, level_value_usd)
        consumed_base = consumed_usd / level.price

        total_base_amount += consumed_base
        total_usd_spent += consumed_usd
        remaining_usd -= consumed_usd
        levels_consumed += 1

    if remaining_usd > 0:
        return {
            "error": "Insufficient liquidity",
            "available_liquidity_usd": total_usd_spent,
            "requested_usd": order_size_usd
        }

    # Calculate metrics
    average_price = total_usd_spent / total_base_amount if total_base_amount > 0 else 0
    slippage_pct = ((average_price - best_price) / best_price) * 100
    price_impact_pct = abs(slippage_pct)

    return {
        "symbol": symbol,
        "exchange": exchange,
        "order_size_usd": order_size_usd,
        "side": side,
        "best_price": best_price,
        "average_execution_price": average_price,
        "slippage_percentage": slippage_pct,
        "price_impact_percentage": price_impact_pct,
        "levels_consumed": levels_consumed,
        "total_base_amount": total_base_amount,
        "sufficient_liquidity": True,
    }


# Tool registry for easy access
AGENT_TOOLS = [
    get_order_book_depth,
    search_trading_pairs,
    get_market_metadata,
    compare_exchanges,
    calculate_market_impact,
]
