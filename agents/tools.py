from typing import Optional, List, Annotated, Any
from pydantic import Field
import asyncio
from pydantic_ai import ModelRetry, RunContext

from data_engine import OrderBook, exchange_manager, analytics
from data_engine.models import LiquidityAnalysis, ExchangeComparison, MarketImpactReport, CrossExchangeComparison, BacktestReport
import ccxt
import logfire


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
    
    TELEMETRY AWARENESS:
    - This tool returns `latency_ms` (network delay) and `circuit_state` (venue health).
    - If `circuit_state` is 'OPEN', data may be stale or recovery mode is active.
    """
    try:
        client = await exchange_manager.get_client(exchange)
        
        # 1. Standardized Health Check
        if client.circuit_breaker.state == "OPEN":
            # Tag the current span for APM visibility
            logfire.error("circuit_breaker_retry", 
                          exchange=exchange, 
                          circuit_state="OPEN", 
                          symbol=symbol)
            raise ModelRetry(
                f"Exchange '{exchange}' is currently suspended due to health issues (Circuit OPEN). "
                f"Please attempt this analysis on a different venue (e.g., 'coinbase' or 'kraken')."
            )

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
            "circuit_state": client.circuit_breaker.state,
            "latency_ms": getattr(client, 'last_request_latency_ms', None)
        }
    except ValueError:
        raise ModelRetry(f"Symbol '{symbol}' not found. Use 'search_trading_pairs' to verify.")


async def compare_exchanges(
    ctx: RunContext[Any],
    symbol: Annotated[str, Field(description="Trading pair to compare across markets (e.g., 'BTC/USDT')")],
    exchanges: Annotated[List[str], Field(description="List of exchanges to compare")] = ["binance", "coinbase", "kraken"],
    order_size: Annotated[float, Field(description="Order size for market impact analysis (optional)")] = 0.0,
    side: Annotated[str, Field(description="Order side for impact analysis: 'buy' or 'sell'")] = "buy",
) -> CrossExchangeComparison | dict:
    """
    Compare liquidity and execution quality across multiple exchanges in parallel.
    
    CRITICAL: This tool performs parallel depth analysis to find the most efficient execution venue.
    
    Use this tool when:
    - User asks "Which exchange has better liquidity for X?"
    - User wants to compare execution costs across venues
    - User asks about arbitrage opportunities
    
    If order_size is provided, performs full market impact analysis with:
    - VWAP calculation for each venue
    - Slippage comparison
    - Fee-adjusted arbitrage detection
    - Circuit breaker health checks
    - Routing recommendation based on lowest total execution cost
    
    If order_size is 0 or not provided, returns basic order book comparison.
    """
    
    # If order_size is provided, use advanced market impact analysis
    if order_size > 0:
        try:
            # Get exchange clients
            clients = []
            for ex in exchanges:
                try:
                    client = await exchange_manager.get_client(ex)
                    clients.append(client)
                except Exception as e:
                    # Skip exchanges that fail to initialize
                    pass
            
            if not clients:
                raise ModelRetry(f"Could not initialize any exchange clients for {exchanges}")
            
            # Use parallel analytics
            comparison = await analytics.compare_liquidity_across_venues(
                symbol=symbol,
                order_size=order_size,
                side=side,
                exchange_clients=clients
            )
            
            return comparison
            
        except Exception as e:
            raise ModelRetry(f"Failed to perform cross-exchange analysis: {str(e)}")
    
    # Otherwise, fall back to basic order book comparison
    # Fetch order books in parallel for performance
    levels = 20  # Default depth for basic comparison
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
                "latency_ms": ob.latency_ms,
                "circuit_state": ob.circuit_state,
                "taker_fee_pct": getattr(ob, 'taker_fee_pct', 0.1)
            }
            for ex, ob in successful_books
        ]
    }





async def get_historical_metrics(
    ctx: RunContext[Any],
    symbol: Annotated[str, Field(description="Trading pair symbol (e.g., 'BTC/USDT')")],
    exchange: Annotated[str, Field(description="Exchange to source data from")] = "binance",
    timeframe: Annotated[str, Field(description="Candle duration")] = "1h",
    lookback_days: Annotated[int, Field(description="Number of days to analyze")] = 1,
) -> dict:
    """
    Analyze historical price and volume trends to identify liquidity conditions.

    Use this tool to detect 'liquidity droughts', volatility spikes, or unusual volume patterns
    that may affect execution strategy.
    
    Returns:
    - Volatility stats (std dev)
    - Volume profile (average, min, max)
    - List of outlier timestamps
    """
    try:
        client = await exchange_manager.get_client(exchange)
        
        # Calculate start timestamp (since)
        now_ms = int(asyncio.get_event_loop().time() * 1000) # Fallback if exchange time not avail
        try:
             now_ms = client.exchange.milliseconds()
        except:
             pass
             
        since = now_ms - (lookback_days * 24 * 60 * 60 * 1000)
        
        ohlcv = await client.fetch_ohlcv(symbol, timeframe=timeframe, since=since)
        
        if not ohlcv:
             return {"error": "No historical data found for this period."}

        # Process Data: [timestamp, open, high, low, close, volume]
        closes = [candle[4] for candle in ohlcv]
        volumes = [candle[5] for candle in ohlcv]
        
        # Calculate Metrics
        import statistics
        avg_price = statistics.mean(closes)
        price_std_dev = statistics.stdev(closes) if len(closes) > 1 else 0
        
        avg_vol = statistics.mean(volumes)
        vol_std_dev = statistics.stdev(volumes) if len(volumes) > 1 else 0
        
        # Outlier Detection
        outliers = []
        for candle in ohlcv:
            ts, _, _, _, _, vol = candle
            if vol > (avg_vol + 2 * vol_std_dev):
                # Convert ms timestamp to readable string
                from datetime import datetime, timezone
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                outliers.append(f"{dt} (Vol: {vol:,.2f})")
        
        # Volume Consistency Score (Simple heuristic)
        # Higher cv (std_dev/mean) -> Lower consistency
        vol_cv = (vol_std_dev / avg_vol) if avg_vol > 0 else 0
        consistency_score = max(1, min(10,int(10 * (1 - min(vol_cv, 1)))))

        return {
            "symbol": symbol,
            "period": f"Last {lookback_days} day(s) ({timeframe} candles)",
            "candle_count": len(ohlcv),
            "price_volatility_std": round(price_std_dev, 4),
            "avg_volume": round(avg_vol, 2),
            "volume_consistency_score": consistency_score, # 1-10
            "outliers": outliers[:5], # Top 5 recent outliers
            "trend_summary": "High volatility" if (price_std_dev / avg_price) > 0.05 else "Stable range"
        }

    except NotImplementedError:
        raise ModelRetry(f"Exchange '{exchange}' does not support historical data fetching.")
    except Exception as e:
        raise ModelRetry(f"Failed to fetch historical data: {str(e)}")



async def calculate_market_impact(
    ctx: RunContext[Any],
    symbol: Annotated[str, Field(description="Trading pair symbol (e.g., 'SOL/USDT')")],
    side: Annotated[str, Field(description="Order side: 'buy' or 'sell'")],
    size: Annotated[float, Field(description="Order size in BASE currency (e.g., 500 SOL)")],
    exchange: Annotated[str, Field(description="Target exchange")] = "binance",
) -> MarketImpactReport:
    """
    Simulate a market order to calculate expected slippage and impact.
    
    Use this tool to answer 'What if' questions like 'What is the impact of selling 1000 SOL?'.
    It 'walks the order book' to determine the Volume Weighted Average Price (VWAP) 
    and compares it to the mid-price.
    
    TELEMETRY AWARENESS:
    - Analyzes `latency_ms` and `circuit_state` during simulation.
    - Triggers CRITICAL risk warnings if `slippage_bps` > 200 bps.
    
    CRITICAL: This tool validates the order against exchange-specific limits (min/max amount, min cost)
    and enforces precision using `exchange.amount_to_precision()`.
    """
    try:
        # 1. Get exchange client
        client = await exchange_manager.get_client(exchange)
        
        # 2. Circuit Breaker Health Check
        if client.circuit_breaker.state == "OPEN":
            logfire.error("circuit_breaker_retry", 
                          exchange=exchange, 
                          circuit_state="OPEN", 
                          symbol=symbol)
            raise ModelRetry(
                f"Exchange '{exchange}' is currently unavailable (circuit breaker OPEN). "
                f"Please try again in a few moments or use a different exchange."
            )
        
        # Markets are already loaded by lifespan preload_exchange
        # No need to call load_markets() again - it's cached
        
        # Get market info for limits (from cache)
        market_limits = None
        try:
            market_info = client.exchange.market(symbol)
            market_limits = market_info.get('limits', {})
        except Exception as e:
            # If market info unavailable, proceed without limits validation
            pass
        
        # 3. Precision Enforcement
        # Ensure the size meets exchange precision before simulation
        try:
            precise_size_str = client.exchange.amount_to_precision(symbol, size)
            precise_size = float(precise_size_str)
            if precise_size != size:
                logfire.info("size_precision_adjusted", original=size, adjusted=precise_size)
            size = precise_size
        except Exception as e:
            # If precision tool fails, proceed with raw size but log it
            logfire.warn("precision_adjustment_failed", error=str(e))
            pass

        # 4. Fetch deep order book
        # Standard fetch is 20. Use 50 for impact to ensure depth coverage.
        limit = 50
        orderbook = await client.fetch_order_book(symbol, limit=limit)
        
        # 4. Run Analytics with limits validation and precision handling
        report = analytics.calculate_market_impact(
            orderbook=orderbook,
            side=side,
            size=size,
            is_quote_size=False,  # Assuming input is always Base amount for now as per docstring
            market_limits=market_limits,
            slippage_threshold_bps=100.0,  # Default threshold: 100 bps (1%)
            exchange=client.exchange  # Pass exchange for precision handling
        )
        
        # 5. Enhanced Error Guidance for Limit Violations
        if report.warning and "below exchange minimum" in report.warning:
            # Extract the minimum from the warning if possible
            raise ModelRetry(
                f"Order validation failed: {report.warning}. "
                f"Please inform the user that their requested order size does not meet exchange requirements. "
                f"Suggest they increase the order size or check the exchange's minimum order limits."
            )
        
        return report
        
    except Exception as e:
        raise ModelRetry(f"Failed to calculate market impact: {str(e)}")


async def run_historical_backtest(
    ctx: RunContext[Any],
    symbol: Annotated[str, Field(description="Trading pair to backtest (e.g., 'SOL/USDT')")],
    order_size: Annotated[float, Field(description="Order size in base currency to simulate")],
    side: Annotated[str, Field(description="Order side: 'buy' or 'sell'")],
    timeframe: Annotated[str, Field(description="Candle timeframe: '1m', '5m', '15m', '1h', '4h', '1d'")],
    lookback_days: Annotated[int, Field(description="Number of days to look back")] = 7,
    exchange: Annotated[str, Field(description="Exchange to backtest on")] = "binance",
) -> BacktestReport:
    """
    Run historical liquidity backtest to simulate order execution during past market conditions.
    
    CRITICAL: This tool performs "time-travel" execution simulation using OHLCV data.
    
    Use this tool when:
    - User asks "How would my trade have performed during [past event]?"
    - User wants to analyze historical slippage patterns
    - User asks about optimal execution windows
    - User wants to compare liquidity during high vs low volatility
    
    The tool:
    1. Fetches historical OHLCV data
    2. Reconstructs synthetic order books from volatility
    3. Simulates order execution at each candle
    4. Returns aggregated metrics (avg/max/min slippage, risk periods)
    
    Returns:
        BacktestReport with comprehensive historical performance analysis
    """
    with logfire.span(
        "historical_backtest",
        symbol=symbol,
        order_size=order_size,
        side=side,
        timeframe=timeframe,
        lookback_days=lookback_days,
        exchange=exchange
    ):
        try:
            # 1. Get exchange client
            with logfire.span("get_exchange_client"):
                client = await exchange_manager.get_client(exchange)
            
            # 2. Circuit Breaker Health Check
            if client.circuit_breaker.state == "OPEN":
                logfire.warn("circuit_breaker_open", exchange=exchange)
                raise ModelRetry(
                    f"Exchange '{exchange}' is currently unavailable (circuit breaker OPEN). "
                    f"Cannot fetch historical data. Please try again later or use a different exchange."
                )
            
            # 3. Capability Guard
            if not client.exchange.has.get('fetchOHLCV', False):
                logfire.error("ohlcv_not_supported", exchange=exchange)
                raise ModelRetry(
                    f"Exchange '{exchange}' does not support historical OHLCV data fetching. "
                    f"Please use an exchange that supports this feature (e.g., Binance, Kraken, Coinbase)."
                )
            
            # 4. Calculate time range
            import time
            from datetime import datetime, timedelta
            
            end_time = int(time.time() * 1000)
            start_time = end_time - (lookback_days * 24 * 60 * 60 * 1000)
            
            # 5. Fetch OHLCV data with rate limit retry
            with logfire.span("fetch_ohlcv", limit=1000) as fetch_span:
                max_retries = 3
                retry_count = 0
                ohlcv_data = None
                
                while retry_count < max_retries:
                    try:
                        ohlcv_data = await client.circuit_breaker.call(
                            client.exchange.fetch_ohlcv,
                            symbol,
                            timeframe,
                            start_time,
                            limit=1000
                        )
                        fetch_span.set_attribute("candles_fetched", len(ohlcv_data) if ohlcv_data else 0)
                        break
                        
                    except ccxt.RateLimitExceeded as e:
                        retry_count += 1
                        logfire.warn("rate_limit_retry", attempt=retry_count, max_retries=max_retries)
                        if retry_count >= max_retries:
                            raise ModelRetry(
                                f"Rate limit exceeded while fetching historical data for {symbol}. "
                                f"The exchange is throttling requests. Please try again in a few minutes."
                            )
                        await client.exchange.sleep(10000)
                        
                    except Exception as e:
                        logfire.error("ohlcv_fetch_failed", error=str(e))
                        raise ModelRetry(f"Failed to fetch OHLCV data: {str(e)}")
            
            if not ohlcv_data or len(ohlcv_data) == 0:
                logfire.warn("no_ohlcv_data", symbol=symbol, timeframe=timeframe)
                raise ModelRetry(
                    f"No historical data available for {symbol} on {exchange} "
                    f"for the requested timeframe ({timeframe}) and lookback period ({lookback_days} days)."
                )
            
            # 6. Get market limits
            with logfire.span("get_market_limits"):
                market_limits = None
                try:
                    market_info = client.exchange.market(symbol)
                    market_limits = market_info.get('limits', {})
                except:
                    pass
            
            # 7. Run backtest simulation
            with logfire.span("simulate_execution", candles=len(ohlcv_data)) as sim_span:
                report = analytics.simulate_historical_execution(
                    ohlcv_data=ohlcv_data,
                    symbol=symbol,
                    order_size=order_size,
                    side=side,
                    exchange=client.exchange,
                    market_limits=market_limits
                )
                
                # Log key metrics
                sim_span.set_attribute("avg_slippage_bps", report.avg_slippage_bps)
                sim_span.set_attribute("max_slippage_bps", report.max_slippage_bps)
                sim_span.set_attribute("high_risk_periods", report.high_risk_periods)
                sim_span.set_attribute("optimal_windows", report.optimal_execution_windows)
            
            # 8. Set timeframe in report
            report.timeframe = timeframe
            
            logfire.info(
                "backtest_complete",
                symbol=symbol,
                candles_analyzed=report.total_candles,
                avg_slippage=report.avg_slippage_bps
            )
            
            return report
            
        except ModelRetry:
            raise
        except Exception as e:
            logfire.error("backtest_failed", error=str(e))
            raise ModelRetry(f"Historical backtest failed: {str(e)}")






# Tool registry for easy access
AGENT_TOOLS = [
    get_order_book_depth,
    search_trading_pairs,
    get_market_metadata,
    compare_exchanges,
    calculate_market_impact,
    get_historical_metrics,
    run_historical_backtest,
]
