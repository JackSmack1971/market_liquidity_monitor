# Market Liquidity Monitor - Features Guide

Complete guide to all features and capabilities of the Market Liquidity Monitor platform.

---

## Table of Contents

1. [Multi-Exchange Comparison](#multi-exchange-comparison)
2. [Historical Backtesting](#historical-backtesting)
3. [Market Impact Analysis](#market-impact-analysis)
4. [Observability & Monitoring](#observability--monitoring)
5. [Circuit Breaker Pattern](#circuit-breaker-pattern)

---

## Multi-Exchange Comparison

### Overview

Compare liquidity across multiple cryptocurrency exchanges in parallel to find the best execution venue and identify arbitrage opportunities.

### How It Works

#### 1. Parallel Order Book Fetching

The system fetches order books from multiple exchanges simultaneously using `asyncio.gather`:

```python
# Fetches from Binance, Kraken, Coinbase in parallel
"Compare SOL liquidity on Binance, Kraken, and Coinbase for a 1000 SOL buy"
```

#### 2. Market Impact Calculation

For each exchange, the system:

- Calculates VWAP (Volume-Weighted Average Price)
- Estimates slippage in basis points
- Applies exchange-specific precision rules
- Validates against minimum order limits

#### 3. Fee-Adjusted Arbitrage Detection

Compares execution costs across venues:

```
Buy Price (Binance):  $142.35 + 0.10% fee = $142.49
Sell Price (Kraken):  $142.80 - 0.16% fee = $142.57
Arbitrage Profit:     $0.08 per SOL (0.056%)
```

#### 4. Circuit Breaker Integration

Automatically excludes exchanges with:

- Open circuit breaker (5+ consecutive failures)
- Recent rate limit violations
- Connection timeouts

### Agent Tool: `compare_exchanges`

**Parameters:**

- `symbol`: Trading pair (e.g., "SOL/USDT")
- `exchanges`: List of exchange IDs (e.g., ["binance", "kraken"])
- `order_size`: Order size in base currency (optional)
- `side`: "buy" or "sell" (optional)
- `levels`: Order book depth (default: 20)

**Returns:**

- `CrossExchangeComparison` model with:
  - Recommended venue
  - Arbitrage opportunity (boolean)
  - Potential profit percentage
  - Per-venue analysis (fill price, slippage, fees)

### Dashboard Visualization

The frontend displays:

- **Arbitrage Alert**: Green banner if profit opportunity detected
- **Routing Recommendation**: Best venue with reasoning
- **Venue Cards**: Side-by-side comparison with metrics
- **Ineligible Venues**: Collapsible section with exclusion reasons

---

## Historical Backtesting

### Overview

Simulate order execution during past market conditions using synthetic order book reconstruction from OHLCV data.

### Methodology

#### 1. Synthetic Order Book Reconstruction

Since historical order book snapshots are not available via unified APIs, we reconstruct them from OHLCV candles:

**Spread Estimation:**

```python
spread_bps = (high - low) / close * 10000
```

**Volatility Modeling (ATR Proxy):**

```python
atr = high - low
volatility_percentile = min(100, (atr / close) * 1000)
```

**Precision Compliance:**

```python
mid_price = exchange.price_to_precision(symbol, close)
```

#### 2. Slippage Simulation

For each historical candle, estimate slippage based on:

```python
size_impact_factor = order_size / volume
volatility_factor = volatility_percentile / 100
estimated_slippage = base_spread + (size_impact * volatility * 100 bps)
```

#### 3. Fill Price Calculation

```python
if side == 'buy':
    fill_price = mid_price * (1 + estimated_slippage / 10000)
else:
    fill_price = mid_price * (1 - estimated_slippage / 10000)
```

#### 4. Risk Period Classification

- **High Risk**: Slippage > 200 bps
- **Optimal**: Slippage < 50 bps
- **Normal**: 50-200 bps

### Agent Tool: `run_historical_backtest`

**Parameters:**

- `symbol`: Trading pair
- `order_size`: Order size in base currency
- `side`: "buy" or "sell"
- `timeframe`: Candle interval ("1m", "5m", "1h", "4h", "1d")
- `lookback_days`: Number of days to analyze (default: 7)
- `exchange`: Exchange ID (default: "binance")

**Returns:**

- `BacktestReport` model with:
  - Average/max/min slippage
  - High-risk period count
  - Optimal execution window count
  - Volatility profile
  - Execution warnings

### Logfire Instrumentation

The backtest tool tracks:

**Spans:**

1. `historical_backtest` (top-level)
2. `get_exchange_client`
3. `fetch_ohlcv` (with candle count)
4. `get_market_limits`
5. `simulate_execution` (with metrics)

**Logged Events:**

- `rate_limit_retry`: Retry attempts during OHLCV fetch
- `no_ohlcv_data`: No data available for timeframe
- `backtest_complete`: Success with key metrics

### Dashboard Visualization

The frontend displays:

- **Metric Cards**: Avg/max slippage, risk periods, optimal windows
- **Volatility Profile**: Avg/max spread, volatility percentile
- **Summary Analysis**: Percentage breakdown of risk vs optimal periods
- **Execution Warnings**: Collapsible list of limit violations

---

## Market Impact Analysis

### Overview

Estimate the price impact of executing a large order on a specific exchange.

### Calculation Method

#### 1. VWAP Calculation

Walk through the order book until the order is filled:

```python
cumulative_volume = 0
cumulative_cost = 0

for level in order_book:
    if cumulative_volume >= order_size:
        break
    volume_at_level = min(level.amount, order_size - cumulative_volume)
    cumulative_cost += volume_at_level * level.price
    cumulative_volume += volume_at_level

vwap = cumulative_cost / cumulative_volume
```

#### 2. Slippage Estimation

```python
mid_price = (best_bid + best_ask) / 2
slippage_bps = abs(vwap - mid_price) / mid_price * 10000
```

#### 3. Severity Classification

- **LOW**: < 10 bps
- **MEDIUM**: 10-50 bps
- **HIGH**: > 50 bps

### Agent Tool: `calculate_market_impact`

**Parameters:**

- `symbol`: Trading pair
- `size`: Order size
- `side`: "buy" or "sell"
- `exchange`: Exchange ID (default: "binance")

**Returns:**

- `MarketImpactReport` model with:
  - Fill price (VWAP)
  - Slippage in bps
  - Severity level
  - Warning (if order violates limits)

---

## Observability & Monitoring

### Logfire Integration

The platform uses Pydantic Logfire for comprehensive observability.

#### Instrumented Components

**1. FastAPI Endpoints**

- Request/response cycles
- Latency tracking
- Error rates

**2. Agent Tools**

- Tool execution time
- Parameter values
- Return data

**3. Historical Backtest**

- OHLCV fetch duration
- Simulation processing time
- Candle count and metrics

#### Viewing Traces

Access the Logfire dashboard to view:

- **Spans**: Hierarchical execution traces
- **Metrics**: Slippage distributions, volatility patterns
- **Errors**: Failed requests with stack traces
- **Performance**: Slow queries and bottlenecks

#### Configuration

Set environment variables:

```bash
LOGFIRE_TOKEN=your_token_here
LOGFIRE_SERVICE_NAME=market-liquidity-monitor
LOGFIRE_ENVIRONMENT=production
```

---

## Circuit Breaker Pattern

### Overview

Prevents cascading failures by temporarily suspending requests to unhealthy exchanges.

### State Machine

```
CLOSED (Healthy)
    ↓ (5 consecutive failures)
OPEN (Suspended)
    ↓ (30 seconds timeout)
HALF_OPEN (Testing)
    ↓ (Success) → CLOSED
    ↓ (Failure) → OPEN
```

### Configuration

- **Failure Threshold**: 5 consecutive errors
- **Timeout**: 30 seconds
- **Half-Open Test**: Single request to verify recovery

### Integration

All agent tools check circuit breaker state:

```python
if client.circuit_breaker.state == "OPEN":
    raise ModelRetry(
        f"Exchange '{exchange}' is currently unavailable. "
        "Please try again later or use a different exchange."
    )
```

### Monitoring

Circuit breaker state is:

- Logged to Logfire on state transitions
- Displayed in dashboard health indicators
- Used to exclude venues from multi-exchange comparison

---

## Advanced Usage Examples

### Example 1: Multi-Exchange Arbitrage Scan

**Query:**

```
"Compare BTC liquidity on Binance, Kraken, and Coinbase for a 10 BTC buy order"
```

**Expected Response:**

- Recommended venue: Binance (lowest slippage)
- Arbitrage opportunity: Yes (0.12% profit)
- Binance: $43,250.50 fill, 8.2 bps slippage
- Kraken: $43,302.10 fill, 19.5 bps slippage
- Coinbase: Circuit breaker OPEN (excluded)

### Example 2: Historical Volatility Analysis

**Query:**

```
"How would selling 1000 ETH have performed during the last 30 days on Binance?"
```

**Expected Response:**

- Avg slippage: 32.5 bps
- Max slippage: 412.8 bps (during volatility spike)
- High-risk periods: 15% of candles
- Optimal windows: 42% of candles
- Avg fill price: $2,245.67

### Example 3: Pre-Trade Risk Assessment

**Query:**

```
"What's the market impact of buying $100,000 worth of SOL on Kraken?"
```

**Expected Response:**

- Fill price: $142.85 (VWAP)
- Slippage: 45.2 bps (MEDIUM severity)
- Order size: 700.35 SOL
- Warning: None (within exchange limits)

---

## Technical Implementation Details

### Data Models

All features use strongly-typed Pydantic models:

- `VenueAnalysis`: Per-exchange liquidity metrics
- `CrossExchangeComparison`: Multi-venue comparison results
- `SyntheticOrderBook`: Reconstructed historical order book
- `BacktestReport`: Historical simulation results
- `MarketImpactReport`: Slippage estimation

### Async Architecture

All I/O operations are async:

- Order book fetching
- OHLCV data retrieval
- Database queries
- Cache operations

### Precision Handling

All prices and amounts respect exchange-specific rules:

```python
price = exchange.price_to_precision(symbol, calculated_price)
amount = exchange.amount_to_precision(symbol, order_size)
```

### Error Handling

Three-tier error strategy:

1. **Circuit Breaker**: Prevents requests to failing exchanges
2. **ModelRetry**: Provides actionable guidance to the agent
3. **Logfire**: Tracks all errors for debugging

---

## Performance Considerations

### Caching Strategy

- **Markets Metadata**: Cached indefinitely (rarely changes)
- **OHLCV Data**: 5-minute TTL
- **Order Books**: No caching (real-time data)

### Rate Limit Management

- **Retry Logic**: 3 attempts with 10-second sleep
- **Circuit Breaker**: Suspends after 5 failures
- **Connection Pooling**: Reuses CCXT clients

### Memory Management

- **Streamlit Cache**: Max 50 entries for historical data
- **Periodic Cleanup**: `gc.collect()` after large backtests
- **On-Demand CSV**: Generated only when user clicks download

---

## Future Enhancements

Planned features for v3.0:

- Real-time WebSocket order book streaming
- Multi-asset portfolio backtesting
- Custom slippage models (ML-based)
- Advanced arbitrage strategies (triangular, cross-exchange)
- Historical order book snapshots (if API support added)

---

For deployment instructions, see [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md).  
For quick start guide, see [QUICKSTART.md](QUICKSTART.md).
