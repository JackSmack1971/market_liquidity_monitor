"""
Analytics engine for advanced liquidity metrics.

Provides pure math functions for:
- VWAP calculation
- Slippage estimation
- Market Impact simulation
"""

from data_engine.models import OrderBook, MarketImpactReport, VenueAnalysis, CrossExchangeComparison, SyntheticOrderBook, BacktestReport
from typing import TYPE_CHECKING, List, Dict, Any
import asyncio
from datetime import datetime
import statistics
import logfire
from config.settings import settings

# Initialize Market Metrics
SLIPPAGE_GAUGE = logfire.metric_gauge(
    "market_slippage_bps",
    unit="bps",
    description="Real-time slippage in basis points"
)
HIGH_RISK_COUNTER = logfire.metric_counter(
    "high_risk_execution_periods_total",
    unit="1",
    description="Total number of execution periods where slippage exceeded 200 bps"
)
ARBITRAGE_PROFIT_GAUGE = logfire.metric_gauge(
    "market_arbitrage_profit_pct",
    unit="%",
    description="Fee-adjusted potential arbitrage profit percentage"
)

if TYPE_CHECKING:
    import ccxt.async_support as ccxt

def calculate_market_impact(
    orderbook: OrderBook,
    side: str,
    size: float,
    is_quote_size: bool = False,
    market_limits: dict = None,
    slippage_threshold_bps: float = 100.0,
    exchange: 'ccxt.Exchange' = None
) -> MarketImpactReport:
    """
    Calculate the market impact (slippage) for a theoretical order.

    Simulates walking the order book depth to fill a specific size.

    Args:
        orderbook: Snapshot of current market depth
        side: 'buy' or 'sell'
        size: Order size
        is_quote_size: If True, 'size' is in Quote currency (e.g., USD). 
                       If False, 'size' is in Base currency (e.g., SOL).
        market_limits: Exchange-specific limits dict from CCXT (optional)
        slippage_threshold_bps: Threshold in bps to trigger high-severity warning (default: 100)
        exchange: CCXT exchange instance for precision handling (optional)

    Returns:
        MarketImpactReport with VWAP and slippage metrics.
    """
    # 1. Determine direction and liquidity source
    # Buy Order -> Consumes Asks (Prices go UP)
    # Sell Order -> Consumes Bids (Prices go DOWN)
    if side.lower() == 'buy':
        levels = orderbook.asks
        reference_price = orderbook.best_ask.price if orderbook.best_ask else 0.0
    else:
        levels = orderbook.bids
        reference_price = orderbook.best_bid.price if orderbook.best_bid else 0.0

    if not levels or reference_price == 0:
        return MarketImpactReport(
            symbol=orderbook.symbol,
            side=side,
            target_size=size,
            target_value_usd=0,
            expected_fill_price=0,
            reference_price=0,
            slippage_bps=0,
            price_impact_percent=0,
            warning="Order book empty"
        )

    # 2. Walk the book
    remaining_size = size
    total_cost = 0.0
    filled_base_amount = 0.0
    
    final_price_level = reference_price

    for level in levels:
        price = level.price
        available_liquidity = level.amount # Base currency volume at this level
        
        # Determine how much we can fill at this level
        if is_quote_size:
            # Size is USD. Convert avail liquidity to USD value approx?
            # Better: convert our remaining USD size to Base approx or just track cost.
            # Strategy: accumulate cost until target USD reached.
            level_value = price * available_liquidity
            
            if level_value >= remaining_size:
                # Fill remaining here
                fill_fraction = remaining_size / price
                total_cost += remaining_size
                filled_base_amount += fill_fraction
                final_price_level = price
                remaining_size = 0
                break
            else:
                # Consume full level
                total_cost += level_value
                filled_base_amount += available_liquidity
                remaining_size -= level_value
                final_price_level = price
        else:
            # Size is Base.
            if available_liquidity >= remaining_size:
                # Fill remaining here
                total_cost += (remaining_size * price)
                filled_base_amount += remaining_size
                final_price_level = price
                remaining_size = 0
                break
            else:
                # Consume full level
                total_cost += (available_liquidity * price)
                filled_base_amount += available_liquidity
                remaining_size -= available_liquidity
                final_price_level = price

    # 3. Market Limits Validation
    warnings = []
    
    # Check if order exceeds available depth
    if remaining_size > 0:
        warnings.append(f"Order exceeds available book depth ({len(levels)} levels fetched). Partial fill simulated.")
    
    # Validate against exchange limits if provided
    if market_limits:
        min_amount = market_limits.get('amount', {}).get('min', 0)
        max_amount = market_limits.get('amount', {}).get('max', float('inf'))
        min_cost = market_limits.get('cost', {}).get('min', 0)
        
        if filled_base_amount < min_amount:
            warnings.append(f"Order size ({filled_base_amount:.8f}) below exchange minimum ({min_amount}). Order would be rejected.")
        
        if filled_base_amount > max_amount:
            warnings.append(f"Order size ({filled_base_amount:.8f}) exceeds exchange maximum ({max_amount}).")
        
        if total_cost < min_cost:
            warnings.append(f"Order value (${total_cost:.2f}) below exchange minimum cost (${min_cost:.2f}). Order would be rejected.")

    # 4. Calculate Metrics
    if filled_base_amount == 0:
         vwap = reference_price # Should not happen unless size 0
    else:
         vwap = total_cost / filled_base_amount

    # Slippage: distance of VWAP from Reference Price
    # Buy: VWAP > Ref (Bad)
    # Sell: VWAP < Ref (Bad)
    if side.lower() == 'buy':
        impact_pct = ((vwap - reference_price) / reference_price) * 100
    else:
        impact_pct = ((reference_price - vwap) / reference_price) * 100
        
    slippage_bps = impact_pct * 100
    
    # 5. Slippage Threshold Alert
    if abs(slippage_bps) > slippage_threshold_bps:
        severity = "HIGH" if abs(slippage_bps) > 200 else "MEDIUM"
        warnings.append(f"⚠️ {severity} SLIPPAGE: {abs(slippage_bps):.2f} bps exceeds threshold ({slippage_threshold_bps} bps). Execution risk elevated.")
    
    # Consolidate warnings
    warning_msg = " | ".join(warnings) if warnings else None
    
    # 6. Apply Exchange Precision (if available)
    if exchange:
        try:
            # Apply precision to amounts and prices
            filled_base_amount = float(exchange.amount_to_precision(orderbook.symbol, filled_base_amount))
            vwap = float(exchange.price_to_precision(orderbook.symbol, vwap))
            final_price_level = float(exchange.price_to_precision(orderbook.symbol, final_price_level))
        except Exception as e:
            # If precision fails, continue with raw values
            pass

    # 7. Record Business Metrics for Observability
    if settings.logfire_token:
        SLIPPAGE_GAUGE.set(abs(slippage_bps), {"symbol": orderbook.symbol, "side": side})
        if abs(slippage_bps) > 200:
             HIGH_RISK_COUNTER.add(1, {"symbol": orderbook.symbol})

    return MarketImpactReport(
        symbol=orderbook.symbol,
        side=side,
        target_size=filled_base_amount, # Actual filled
        target_value_usd=total_cost,
        expected_fill_price=round(vwap, 8),
        reference_price=reference_price,
        slippage_bps=round(slippage_bps, 2),
        price_impact_percent=round(impact_pct, 4),
        critical_depth_level=final_price_level,
        warning=warning_msg,
        latency_ms=orderbook.latency_ms
    )


async def compare_liquidity_across_venues(
    symbol: str,
    order_size: float,
    side: str,
    exchange_clients: List[Any],  # List of ExchangeClient instances
    slippage_threshold_bps: float = 100.0
) -> CrossExchangeComparison:
    """
    Compare liquidity and execution quality across multiple exchanges in parallel.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        order_size: Order size in base currency
        side: 'buy' or 'sell'
        exchange_clients: List of ExchangeClient instances
        slippage_threshold_bps: Threshold for slippage warnings
    
    Returns:
        CrossExchangeComparison with routing recommendation and arbitrage analysis
    """
    
    async def analyze_venue(client) -> VenueAnalysis:
        """Analyze a single venue."""
        exchange_id = client.exchange_id
        
        # Check circuit breaker state
        circuit_state = client.circuit_breaker.state
        
        # If circuit is OPEN, mark as ineligible
        if circuit_state == "OPEN":
            return VenueAnalysis(
                exchange_id=exchange_id,
                fill_price=0.0,
                slippage_bps=0.0,
                execution_warnings=[f"Exchange unavailable (circuit {circuit_state})"],
                circuit_state=circuit_state,
                is_eligible=False,
                taker_fee_pct=0.1
            )
        
        try:
            # Fetch order book
            orderbook = await client.fetch_order_book(symbol, limit=50)
            
            # Get market limits
            market_limits = None
            try:
                market_info = client.exchange.market(symbol)
                market_limits = market_info.get('limits', {})
            except:
                pass
            
            # Calculate market impact
            impact_report = calculate_market_impact(
                orderbook=orderbook,
                side=side,
                size=order_size,
                market_limits=market_limits,
                slippage_threshold_bps=slippage_threshold_bps,
                exchange=client.exchange
            )
            
            # Check eligibility
            is_eligible = True
            warnings = []
            
            if impact_report.warning:
                warnings.append(impact_report.warning)
                if "below exchange minimum" in impact_report.warning:
                    is_eligible = False
            
            # Get taker fee (default 0.1% if not available)
            taker_fee_pct = 0.1
            try:
                if hasattr(client.exchange, 'fees') and 'trading' in client.exchange.fees:
                    taker_fee_pct = client.exchange.fees['trading'].get('taker', 0.001) * 100
            except:
                pass
            
            return VenueAnalysis(
                exchange_id=exchange_id,
                fill_price=impact_report.expected_fill_price,
                slippage_bps=impact_report.slippage_bps,
                execution_warnings=warnings,
                circuit_state=circuit_state,
                is_eligible=is_eligible,
                taker_fee_pct=taker_fee_pct
            )
            
        except Exception as e:
            # Mark as ineligible on error
            return VenueAnalysis(
                exchange_id=exchange_id,
                fill_price=0.0,
                slippage_bps=0.0,
                execution_warnings=[f"Analysis failed: {str(e)}"],
                circuit_state=circuit_state,
                is_eligible=False,
                taker_fee_pct=0.1
            )
    
    # Parallel execution
    venue_analyses = await asyncio.gather(
        *[analyze_venue(client) for client in exchange_clients],
        return_exceptions=False
    )
    
    # Filter eligible venues
    eligible_venues = [v for v in venue_analyses if v.is_eligible]
    
    if not eligible_venues:
        # No eligible venues
        return CrossExchangeComparison(
            symbol=symbol,
            order_size=order_size,
            side=side,
            recommended_venue="None",
            arbitrage_opportunity=False,
            venue_analyses=venue_analyses,
            comparison_summary="No eligible venues found for this order size."
        )
    
    # Find best venue (lowest slippage)
    best_venue = min(eligible_venues, key=lambda v: abs(v.slippage_bps))
    
    # Arbitrage detection (for buy orders: lowest ask vs highest bid)
    arbitrage_opportunity = False
    potential_profit_pct = None
    
    if len(eligible_venues) >= 2 and side == 'buy':
        # Find lowest buy price and highest sell price
        buy_prices = [(v.exchange_id, v.fill_price, v.taker_fee_pct) for v in eligible_venues]
        
        lowest_buy = min(buy_prices, key=lambda x: x[1])
        highest_sell = max(buy_prices, key=lambda x: x[1])
        
        # Fee-adjusted profit
        buy_cost = lowest_buy[1] * (1 + lowest_buy[2] / 100)
        sell_revenue = highest_sell[1] * (1 - highest_sell[2] / 100)
        
        if sell_revenue > buy_cost:
            arbitrage_opportunity = True
            potential_profit_pct = ((sell_revenue - buy_cost) / buy_cost) * 100
            
            # Record Arbitrage Metric
            if settings.logfire_token:
                ARBITRAGE_PROFIT_GAUGE.set(
                    potential_profit_pct, 
                    {
                        "symbol": symbol, 
                        "venue_pair": f"{lowest_buy[0]}-{highest_sell[0]}"
                    }
                )
    
    # Generate summary with fee-adjusted insights
    summary_parts = [
        f"Recommended venue: {best_venue.exchange_id} (Price: {best_venue.fill_price:.2f}, Slippage: {best_venue.slippage_bps:.2f} bps, Fee: {best_venue.taker_fee_pct:.3f}%)"
    ]
    
    if arbitrage_opportunity:
        summary_parts.append(f"🔥 Arbitrage opportunity detected: {potential_profit_pct:.2f}% potential profit (fee-adjusted)")
    
    if len(eligible_venues) < len(venue_analyses):
        ineligible_count = len(venue_analyses) - len(eligible_venues)
        summary_parts.append(f"⚠️ {ineligible_count} venue(s) excluded due to health or limit issues")
    
    comparison_summary = ". ".join(summary_parts)
    
    return CrossExchangeComparison(
        symbol=symbol,
        order_size=order_size,
        side=side,
        recommended_venue=best_venue.exchange_id,
        arbitrage_opportunity=arbitrage_opportunity,
        potential_profit_pct=potential_profit_pct,
        venue_analyses=venue_analyses,
        comparison_summary=comparison_summary
    )


def reconstruct_synthetic_book(
    candle: List[float],  # [timestamp, open, high, low, close, volume]
    symbol: str,
    exchange: 'ccxt.Exchange' = None,
    depth_levels: int = 10
) -> SyntheticOrderBook:
    """
    Reconstruct a synthetic order book from OHLCV candle data.
    
    Uses volatility-based spread estimation and ATR modeling for depth.
    
    Args:
        candle: OHLCV data [timestamp, open, high, low, close, volume]
        symbol: Trading pair
        exchange: CCXT exchange instance for precision (optional)
        depth_levels: Number of synthetic depth levels to generate
    
    Returns:
        SyntheticOrderBook with estimated spread and depth
    """
    timestamp_ms, open_price, high, low, close, volume = candle
    timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
    
    # Calculate volatility-based spread
    # Spread estimation: (high - low) / close * 10000 bps
    price_range = high - low
    spread_bps = (price_range / close) * 10000 if close > 0 else 0
    
    # Calculate ATR (Average True Range) proxy
    # For single candle, use high-low range
    atr = price_range
    
    # Volatility percentile (normalized 0-100)
    # Higher range relative to close = higher volatility
    volatility_percentile = min(100, (atr / close) * 1000) if close > 0 else 0
    
    # Apply exchange precision if available
    mid_price = close
    if exchange:
        try:
            mid_price = float(exchange.price_to_precision(symbol, close))
            spread_bps = float(exchange.price_to_precision(symbol, spread_bps))
        except:
            pass
    
    return SyntheticOrderBook(
        timestamp=timestamp,
        symbol=symbol,
        mid_price=mid_price,
        estimated_spread_bps=spread_bps,
        volatility_percentile=volatility_percentile,
        synthetic_depth_levels=depth_levels,
        high=high,
        low=low,
        volume=volume
    )


def simulate_historical_execution(
    ohlcv_data: List[List[float]],  # List of candles
    symbol: str,
    order_size: float,
    side: str,
    exchange: 'ccxt.Exchange' = None,
    market_limits: dict = None
) -> BacktestReport:
    """
    Simulate order execution across historical candles.
    
    Args:
        ohlcv_data: List of OHLCV candles
        symbol: Trading pair
        order_size: Order size in base currency
        side: 'buy' or 'sell'
        exchange: CCXT exchange instance for precision
        market_limits: Exchange limits for validation
    
    Returns:
        BacktestReport with aggregated metrics
    """
    if not ohlcv_data:
        raise ValueError("No OHLCV data provided for backtest")
    
    slippages = []
    fill_prices = []
    warnings = []
    high_risk_count = 0
    optimal_count = 0
    
    # Extract timestamps for date range
    start_timestamp = ohlcv_data[0][0]
    end_timestamp = ohlcv_data[-1][0]
    
    for candle in ohlcv_data:
        # Reconstruct synthetic book
        synthetic_book = reconstruct_synthetic_book(candle, symbol, exchange)
        
        # Estimate slippage based on spread and order size
        # Larger orders relative to volume = higher slippage
        mid_price = synthetic_book.mid_price
        spread_bps = synthetic_book.estimated_spread_bps
        volume = synthetic_book.volume
        
        # Slippage model: base spread + size impact
        # Size impact: (order_size / volume) * volatility_factor
        size_impact_factor = (order_size / volume) if volume > 0 else 1.0
        volatility_factor = synthetic_book.volatility_percentile / 100
        
        # Total slippage = spread + (size_impact * volatility * 100 bps)
        estimated_slippage_bps = spread_bps + (size_impact_factor * volatility_factor * 100)
        
        # Calculate fill price
        if side.lower() == 'buy':
            # Buy: pay spread above mid
            fill_price = mid_price * (1 + estimated_slippage_bps / 10000)
        else:
            # Sell: receive spread below mid
            fill_price = mid_price * (1 - estimated_slippage_bps / 10000)
        
        # Apply precision
        if exchange:
            try:
                fill_price = float(exchange.price_to_precision(symbol, fill_price))
            except:
                pass
        
        slippages.append(estimated_slippage_bps)
        fill_prices.append(fill_price)
        
        # Track risk periods
        if estimated_slippage_bps > 200:
            high_risk_count += 1
        elif estimated_slippage_bps < 50:
            optimal_count += 1
        
        # Validate against limits
        if market_limits:
            min_amount = market_limits.get('amount', {}).get('min', 0)
            min_cost = market_limits.get('cost', {}).get('min', 0)
            
            if order_size < min_amount:
                warnings.append(f"Order size {order_size} below minimum {min_amount} at {synthetic_book.timestamp}")
            
            total_cost = order_size * fill_price
            if total_cost < min_cost:
                warnings.append(f"Order cost ${total_cost:.2f} below minimum ${min_cost:.2f} at {synthetic_book.timestamp}")
    
    # Aggregate metrics
    avg_slippage = statistics.mean(slippages) if slippages else 0
    max_slippage = max(slippages) if slippages else 0
    min_slippage = min(slippages) if slippages else 0
    avg_fill_price = statistics.mean(fill_prices) if fill_prices else 0
    
    # Calculate total cost
    theoretical_total_cost = sum(fill_prices) * order_size if fill_prices else 0
    
    # Volatility profile
    volatility_profile = {
        "avg_spread_bps": statistics.mean([reconstruct_synthetic_book(c, symbol).estimated_spread_bps for c in ohlcv_data]),
        "max_spread_bps": max([reconstruct_synthetic_book(c, symbol).estimated_spread_bps for c in ohlcv_data]),
        "avg_volatility_percentile": statistics.mean([reconstruct_synthetic_book(c, symbol).volatility_percentile for c in ohlcv_data])
    }
    
    return BacktestReport(
        symbol=symbol,
        exchange=exchange.id if exchange else "unknown",
        timeframe="unknown",  # Will be set by caller
        start_date=datetime.fromtimestamp(start_timestamp / 1000),
        end_date=datetime.fromtimestamp(end_timestamp / 1000),
        order_size=order_size,
        side=side,
        total_candles=len(ohlcv_data),
        avg_slippage_bps=avg_slippage,
        max_slippage_bps=max_slippage,
        min_slippage_bps=min_slippage,
        avg_fill_price=avg_fill_price,
        theoretical_total_cost=theoretical_total_cost,
        execution_warnings=list(set(warnings)),  # Deduplicate
        volatility_profile=volatility_profile,
        high_risk_periods=high_risk_count,
        optimal_execution_windows=optimal_count
    )
