"""
Analytics engine for advanced liquidity metrics.

Provides pure math functions for:
- VWAP calculation
- Slippage estimation
- Market Impact simulation
"""

from data_engine.models import OrderBook, MarketImpactReport

def calculate_market_impact(
    orderbook: OrderBook,
    side: str,
    size: float,
    is_quote_size: bool = False,
    market_limits: dict = None,
    slippage_threshold_bps: float = 100.0
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
        warning=warning_msg
    )
