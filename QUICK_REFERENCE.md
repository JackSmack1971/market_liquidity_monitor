# Quick Reference Guide - Enhanced Features

## 🚀 Quick Start

### Launch Enhanced System
```bash
# 1. Start Redis (optional)
docker run -d -p 6379:6379 redis:alpine

# 2. Launch enhanced frontend
streamlit run market_liquidity_monitor/frontend/enhanced_app.py

# 3. (Optional) Start API server
uvicorn market_liquidity_monitor.api.main:app --reload
```

---

## 📊 Common Tasks

### Compare Exchanges
```python
from market_liquidity_monitor.agents.tools import compare_exchanges

result = await compare_exchanges(
    symbol="SOL/USDT",
    exchanges=["binance", "coinbase", "kraken"]
)

print(f"Best exchange: {result['tightest_spread_exchange']}")
if result.get('arbitrage_opportunity_pct'):
    print(f"Arbitrage: {result['arbitrage_route']}")
```

### Calculate Market Impact
```python
from market_liquidity_monitor.agents.tools import calculate_market_impact

impact = await calculate_market_impact(
    symbol="BTC/USDT",
    order_size_usd=50000,
    side="buy",
    exchange="binance"
)

print(f"Slippage: {impact['slippage_percentage']:.3f}%")
print(f"Avg price: ${impact['average_execution_price']:.2f}")
```

### Track Historical Data
```python
from market_liquidity_monitor.data_engine.historical import historical_tracker

# Capture snapshot
snapshot = await historical_tracker.capture_snapshot("SOL/USDT", "binance")

# Get historical data
snapshots = await historical_tracker.get_snapshots(
    symbol="SOL/USDT",
    exchange="binance",
    hours=24
)

# Calculate baseline
baseline = await historical_tracker.get_baseline_metrics(
    symbol="SOL/USDT",
    exchange="binance"
)
```

### Check for Alerts
```python
alerts = await historical_tracker.detect_anomalies(
    symbol="SOL/USDT",
    exchange="binance",
    threshold_pct=30.0
)

for alert in alerts:
    print(f"{alert.severity}: {alert.message}")
```

### Stream Real-Time Data
```python
from market_liquidity_monitor.data_engine.websocket_stream import OrderBookStream

stream = OrderBookStream("binance")

async def handle_update(orderbook):
    print(f"Spread: {orderbook.spread_percentage:.3f}%")

await stream.subscribe("SOL/USDT", handle_update)
await stream.start()
```

### Use Caching
```python
from market_liquidity_monitor.data_engine.cache import cache_manager

# Connect
await cache_manager.connect()

# Cache order book
await cache_manager.cache_orderbook("SOL/USDT", "binance", orderbook, ttl=5)

# Retrieve
cached = await cache_manager.get_orderbook("SOL/USDT", "binance")

# Stats
stats = await cache_manager.get_stats()
print(f"Cache hits: {stats.get('redis_hits', 0)}")
```

---

## 🎨 Visualizations

### Liquidity Heatmap
```python
from market_liquidity_monitor.frontend.advanced_visualizations import create_liquidity_heatmap

fig = create_liquidity_heatmap(orderbook, levels=30)
st.plotly_chart(fig)
```

### Exchange Comparison Chart
```python
from market_liquidity_monitor.frontend.advanced_visualizations import create_exchange_comparison_chart

comparison = await compare_exchanges("SOL/USDT", ["binance", "coinbase"])
fig = create_exchange_comparison_chart(comparison)
st.plotly_chart(fig)
```

### Historical Trend Chart
```python
from market_liquidity_monitor.frontend.advanced_visualizations import create_historical_trend_chart

snapshots = await historical_tracker.get_snapshots("SOL/USDT", "binance", hours=24)
fig = create_historical_trend_chart(snapshots)
st.plotly_chart(fig)
```

---

## 🔌 API Endpoints

### Multi-Exchange Comparison
```bash
POST /api/v1/compare-exchanges?symbol=SOL/USDT&exchanges=binance&exchanges=coinbase
```

### Market Impact
```bash
POST /api/v1/market-impact?symbol=SOL/USDT&order_size_usd=10000&side=buy
```

### Order Book
```bash
GET /api/v1/orderbook/SOL/USDT?exchange=binance&levels=20
```

### Analyze with LLM
```bash
POST /api/v1/analyze
Content-Type: application/json

{
  "query": "How is SOL liquidity?",
  "symbol": "SOL/USDT",
  "exchange": "binance"
}
```

---

## ⚙️ Configuration

### Environment Variables
```bash
# OpenRouter API
OPENROUTER_API_KEY=your_key_here
OPENROUTER_API_BASE=https://openrouter.ai/api/v1

# Exchange (optional)
DEFAULT_EXCHANGE=binance

# Redis (optional)
REDIS_URL=redis://localhost:6379
CACHE_DEFAULT_TTL=60

# Model
DEFAULT_MODEL=anthropic/claude-3.5-sonnet
```

### Settings in Code
```python
from market_liquidity_monitor.config import settings

print(settings.default_exchange)  # binance
print(settings.default_model)     # anthropic/claude-3.5-sonnet
```

---

## 🧪 Testing

### Run All Tests
```bash
pytest market_liquidity_monitor/tests/ -v
```

### Run Specific Test
```bash
pytest market_liquidity_monitor/tests/test_advanced_features.py::TestMultiExchangeComparison::test_compare_exchanges_success -v
```

### Coverage
```bash
pytest --cov=market_liquidity_monitor --cov-report=term-missing
```

---

## 🐛 Debugging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Cache Stats
```python
stats = await cache_manager.get_stats()
print(stats)
```

### Verify Exchange Connection
```python
from market_liquidity_monitor.data_engine import exchange_manager

client = await exchange_manager.get_client("binance")
markets = await client.fetch_markets()
print(f"Connected to {len(markets)} markets")
```

---

## 📈 Performance Tips

1. **Use Caching**: Enable Redis for 100x performance boost
2. **Batch Requests**: Compare multiple exchanges in parallel
3. **WebSocket Streaming**: Use for real-time updates (lower latency than polling)
4. **Adjust TTL**: Increase cache TTL for less volatile data
5. **Limit Levels**: Fetch fewer order book levels when deep analysis not needed

### Example: Optimized Multi-Exchange Query
```python
# Good: Parallel execution with caching
result = await compare_exchanges("SOL/USDT", ["binance", "coinbase", "kraken"])
await cache_manager.cache_comparison("SOL/USDT", exchanges, result, ttl=10)

# Bad: Sequential fetches without caching
for exchange in ["binance", "coinbase", "kraken"]:
    client = await exchange_manager.get_client(exchange)
    orderbook = await client.fetch_order_book("SOL/USDT")
```

---

## 🔐 Security Best Practices

1. **Never commit API keys**: Use `.env` file
2. **Read-only mode**: System only reads market data (no trading)
3. **Rate limiting**: CCXT handles exchange rate limits
4. **Input validation**: All inputs validated via Pydantic
5. **CORS**: Configure allowed origins in settings

---

## 📦 Dependencies

### Required
- `fastapi>=0.104.0` - API framework
- `pydantic>=2.5.0` - Data validation
- `pydantic-ai>=0.0.13` - LLM integration
- `ccxt>=4.2.0` - Exchange APIs
- `streamlit>=1.29.0` - Frontend
- `plotly>=5.18.0` - Visualizations

### Optional
- `redis>=5.0.0` - Caching (falls back to in-memory)

### Install All
```bash
pip install -r market_liquidity_monitor/requirements.txt
```

---

## 🆘 Common Issues

### "Redis connection failed"
**Cause**: Redis not running
**Solution**: System works with in-memory cache. To use Redis:
```bash
docker run -d -p 6379:6379 redis:alpine
```

### "No historical data"
**Cause**: No snapshots captured yet
**Solution**:
```python
await historical_tracker.capture_snapshot("SOL/USDT", "binance")
```

### "Exchange not supported"
**Cause**: Exchange doesn't exist in CCXT
**Solution**: Use supported exchanges:
```python
import ccxt
print(ccxt.exchanges)  # List all supported exchanges
```

### "Rate limit exceeded"
**Cause**: Too many requests
**Solution**: Enable rate limiting:
```python
# settings.py
enable_rate_limit = True  # CCXT will handle throttling
```

---

## 🎯 Use Case Examples

### 1. Find Best Exchange for Large Order
```python
# Compare 5 exchanges
result = await compare_exchanges(
    symbol="BTC/USDT",
    exchanges=["binance", "coinbase", "kraken", "bybit", "okx"]
)

print(f"Deepest liquidity: {result['deepest_liquidity_exchange']}")
print(f"Tightest spread: {result['tightest_spread_exchange']}")

# Calculate impact on best exchange
impact = await calculate_market_impact(
    symbol="BTC/USDT",
    order_size_usd=100000,
    side="buy",
    exchange=result['deepest_liquidity_exchange']
)

if impact['slippage_percentage'] < 0.5:
    print("✅ Acceptable slippage")
else:
    print("❌ High slippage - reduce order size")
```

### 2. Monitor for Arbitrage Opportunities
```python
from market_liquidity_monitor.data_engine.websocket_stream import MultiExchangeStream

stream = MultiExchangeStream(["binance", "coinbase", "kraken"])

async def check_arbitrage(data):
    if len(data) < 2:
        return

    prices = {ex: ob.best_bid.price for ex, ob in data.items() if ob.best_bid}
    best_bid_ex = max(prices, key=prices.get)
    best_bid = prices[best_bid_ex]

    prices = {ex: ob.best_ask.price for ex, ob in data.items() if ob.best_ask}
    best_ask_ex = min(prices, key=prices.get)
    best_ask = prices[best_ask_ex]

    if best_bid > best_ask:
        profit_pct = ((best_bid - best_ask) / best_ask) * 100
        print(f"⚡ ARBITRAGE: {profit_pct:.2f}% - Buy {best_ask_ex}, Sell {best_bid_ex}")

await stream.subscribe("BTC/USDT", check_arbitrage)
await stream.start()
```

### 3. Track Liquidity Health Over Time
```python
# Start continuous tracking
asyncio.create_task(
    historical_tracker.start_continuous_tracking(
        symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        exchange="binance",
        interval_seconds=60
    )
)

# Check for anomalies every 5 minutes
while True:
    alerts = await historical_tracker.detect_anomalies("BTC/USDT", "binance")
    if alerts:
        for alert in alerts:
            print(f"🚨 {alert.severity}: {alert.message}")
    await asyncio.sleep(300)
```

---

**Version**: 2.0.0
**Last Updated**: 2026-01-02
