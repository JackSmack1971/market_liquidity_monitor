from typing import Optional, List, Annotated, Any
from pydantic import Field
import asyncio
from pydantic_ai import ModelRetry, RunContext

from data_engine import OrderBook, exchange_manager
from data_engine.models import LiquidityAnalysis, ExchangeComparison


async def get_order_book_depth(
    ctx: RunContext[Any],
    symbol: Annotated[str, Field(description="Standard trading pair symbol (e.g., 'BTC/USDT', 'SOL/USDC')")],
    exchange: Annotated[str, Field(description="Target cryptocurrency exchange")] = "binance",
    levels: Annotated[int, Field(description="Depth of the book to retrieve")] = 20,
) -> OrderBook:
    """
    Fetch real-time order book depth for a specific trading pair.

    Use this tool when you need to analyze the current supply (asks) and demand (bids). 
    It is essential for calculating spreads, market depth, and identifying buy/sell walls.
    If the ticker symbol is ambiguous, use search_trading_pairs first to find the exact symbol format (e.g., 'BTC/USDT').
    """
    try:
        client = await exchange_manager.get_client(exchange)
        orderbook = await client.fetch_order_book(symbol, limit=levels)
        return orderbook
    except ValueError as e:
        # If the symbol doesn't exist, guide the model to search for it
        raise ModelRetry(
            f"The symbol '{symbol}' was not found on '{exchange}'. Please use 'search_trading_pairs' to find the exact symbol format for this exchange."
        )
    except Exception as e:
        if "exchange not found" in str(e).lower():
            raise ModelRetry(f"Exchange '{exchange}' is not supported. Try 'binance', 'coinbase', or 'kraken'.")
        raise e


async def search_trading_pairs(
    ctx: RunContext[Any],
    query: Annotated[str, Field(description="The coin or token to find pairs for (e.g., 'SOL', 'PEPE')")],
    exchange: Annotated[str, Field(description="Exchange to search within")] = "binance",
) -> list[str]:
    """
    Search for all available trading pairs matching a specific token or term.

    MANDATORY: Use this tool if you are unsure of the exact symbol formatting on a specific exchange. 
    Common formats include 'BTC/USDT', 'BTC/USD', or 'BTC-PERP'.
    This prevents 'Symbol not found' errors in other tools.
    """
    try:
        client = await exchange_manager.get_client(exchange)
        symbols = await client.search_symbol(query)
        if not symbols:
            raise ModelRetry(f"No trading pairs found for '{query}' on '{exchange}'. Try a different search term or exchange.")
        return symbols
    except Exception as e:
        if "exchange not found" in str(e).lower():
            raise ModelRetry(f"Exchange '{exchange}' is not supported.")
        raise e


async def get_market_metadata(
    ctx: RunContext[Any],
    symbol: Annotated[str, Field(description="Precision trading pair symbol")],
    exchange: Annotated[str, Field(description="Target exchange")] = "binance",
) -> dict:
    """
    Retrieve critical market metadata for a trading pair.

    MANDATORY: Use this tool before suggesting any order sizes or executing simulations. 
    It provides:
    - Minimum order size (amount)
    - Minimum order cost (price * amount) - CRITICAL for avoiding exchange rejections.
    - Price and amount precision requirements.
    Ensure your suggested sizes meet these limits to prevent 'insufficient order size' errors.
    """
    try:
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
    except ValueError:
        raise ModelRetry(f"Symbol '{symbol}' not found. Use 'search_trading_pairs' to verify.")


async def compare_exchanges(
    ctx: RunContext[Any],
    symbol: Annotated[str, Field(description="Trading pair to compare across markets (e.g., 'BTC/USDT')")],
    exchanges: Annotated[List[str], Field(description="List of exchanges to compare")] = ["binance", "coinbase", "kraken"],
    levels: Annotated[int, Field(description="Amount of depth to compare.")] = 20,
) -> dict:
    """
    Compare liquidity and pricing for a single symbol across multiple exchanges in parallel.

    Use this tool to identify:
    - Where the tightest spread is currently located
    - Arbitrage opportunities (price discrepancies between markets)
    - Which exchange has the deepest 'walls' or support levels
    Returns localized metrics for each exchange plus a comparative synthesis.
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
        raise ModelRetry(f"Could not fetch data for '{symbol}' from any of the requested exchanges: {exchanges}. Verify the symbol exists on these exchanges.")

    # Calculate comparative metrics
    exchange_names = [ex for ex, _ in successful_books]
    
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
        arbitrage_profit_pct = ((best_bid_data[1] - best_ask_data[1]) / best_ask_data[1]) * 100
        arbitrage_route = f"Buy on {best_ask_exchange} @ ${best_ask_data[1]:.2f}, Sell on {best_bid_exchange} @ ${best_bid_data[1]:.2f}"

    # Calculate average spread
    valid_spreads = [s for _, s in spreads if s != float('inf')]
    avg_spread = sum(valid_spreads) / len(valid_spreads) if valid_spreads else 0

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
        "total_liquidity_volume": sum(depth for _, depth in depths),
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
    ctx: RunContext[Any],
    symbol: Annotated[str, Field(description="Target trading pair symbol")],
    order_size_usd: Annotated[float, Field(description="Simulated order size in USD")],
    side: Annotated[str, Field(description="Trade direction ('buy' or 'sell')")] = "buy",
    exchange: Annotated[str, Field(description="Exchange to simulate on")] = "binance",
) -> dict:
    """
    Simulate execution of a specific order size to determine slippage and price impact.

    Use this tool to answer questions like 'How much slippage for a $10k trade?' or 'Can I buy 5 BTC without moving the price?'.
    It validates your order against exchange limits (amount and cost) and walks through the actual limit orders in the book to provide a realistic execution estimate.
    If the order is too small for the exchange, This tool will provide feedback via a retry request.
    """
    try:
        client = await exchange_manager.get_client(exchange)
        orderbook = await client.fetch_order_book(symbol, limit=100)
    except Exception:
        raise ModelRetry(f"Failed to fetch order book for '{symbol}' on '{exchange}'. Please verify parameters.")

    # Determine which side of the book to use
    levels = orderbook.asks if side == "buy" else orderbook.bids
    best_price = orderbook.best_ask.price if side == "buy" else orderbook.best_bid.price

    if not levels or not best_price:
        raise ModelRetry(f"Insufficient liquidity data to simulate {side} order for {symbol} on {exchange}.")

    # Validate order limits before simulation
    # Convert USD size to base amount for validation
    approx_amount = order_size_usd / best_price
    is_valid, error_msg = client.validate_order_limits(symbol, approx_amount, best_price)
    if not is_valid:
        raise ModelRetry(
            f"The proposed order does not meet {exchange} requirements: {error_msg}. "
            "Please adjust the order size or try a more liquid pair."
        )

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
            "requested_usd": order_size_usd,
            "sufficient_liquidity": False
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
