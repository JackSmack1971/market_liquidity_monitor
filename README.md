# Market Liquidity Monitor

A production-grade system that combines real-time market data with LLM reasoning to monitor liquidity, analyze order book depth, and simulate historical execution.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface (Streamlit)                │
│  - Chat interface for natural language queries               │
│  - Real-time order book visualization                        │
│  - Multi-exchange comparison dashboard                       │
│  - Historical backtest results viewer                        │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                         │
│  - Async API endpoints with Logfire observability           │
│  - Circuit breaker pattern for resilience                   │
│  - Dependency injection for exchange/agent management        │
└────────────┬────────────────────────────────────────────────┘
             │
             ├───────────┬─────────────┬──────────────┬────────┐
             ▼           ▼             ▼              ▼        ▼
┌──────────────┐  ┌────────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐
│ Data Engine  │  │ LLM Agent  │  │  Redis  │  │PostgreSQL│  │ Logfire  │
│   (CCXT)     │  │(Pydantic-AI)│  │  Cache  │  │   DB     │  │   APM    │
│              │  │             │  │         │  │          │  │          │
│ Multi-Venue  │  │ 6 Tools:    │  │ 5min TTL│  │ History  │  │ Spans &  │
│ Parallel     │  │ - Backtest  │  │ Markets │  │ Snapshots│  │ Metrics  │
│ Execution    │  │ - Compare   │  │ OHLCV   │  │ Alerts   │  │ Tracking │
│              │  │ - Impact    │  │         │  │          │  │          │
└──────────────┘  └────────────┘  └─────────┘  └──────────┘  └──────────┘
```

## Components

### 1. Data Engine (CCXT)

- **Async Order Book Fetching**: Non-blocking market data retrieval
- **Multi-Exchange Support**: Unified interface across exchanges
- **Precision Handling**: Exchange-specific decimal requirements

### 2. Reasoning Agent (Pydantic-AI + OpenRouter)

- **Tool Calling**: LLM can invoke market data functions
- **Synthesized Analysis**: Beyond raw numbers to insights
- **Type-Safe Outputs**: Structured responses via Pydantic models

### 3. Backend (FastAPI)

- **Async Execution**: Responsive server architecture
- **Dependency Injection**: Clean separation of concerns
- **Security**: Environment-based secrets management

### 4. Frontend (Streamlit)

- **Chat Interface**: Natural language queries
- **Real-time Visualization**: Order book depth charts
- **Session Management**: Conversation history

### 5. Advanced Features

- **Historical Analysis**: OHLCV trend tracking with volatility metrics
- **Liquidity Alerts**: Hybrid detection (Depth + Trend) for anomalies
- **Market Impact**: Simulate order slippage scaling
- **System Health**: Circuit Breaker status and connection monitoring
- **Multi-Exchange Comparison**: Parallel liquidity analysis with arbitrage detection (NEW)
- **Historical Backtesting**: Time-travel execution simulation with synthetic order books (NEW)
- **Docker Deployment**: Production-grade containerization (see [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md))

### 6. Resilience & Fault Tolerance

- **Circuit Breaker**: Auto-suspends failing exchanges (Threshold: 5, Timeout: 30s)
- **Connection Pooling**: Reuses CCXT clients for rate limit stability
- **Caching**: 5-minute TTL for historical data

## New Features (v2.0)

### Multi-Exchange Comparison

Compare liquidity across multiple exchanges in parallel with intelligent routing recommendations.

**Key Capabilities:**

- **Parallel Analysis**: Fetches order books from multiple venues simultaneously
- **Arbitrage Detection**: Identifies fee-adjusted profit opportunities
- **Venue Routing**: Recommends optimal exchange for execution
- **Circuit Breaker Aware**: Excludes unhealthy exchanges automatically

**Example Query:**

```
"Compare SOL liquidity on Binance and Kraken for a 1000 SOL buy order"
```

**Agent Response:**

- Best venue recommendation with reasoning
- Fill price (VWAP) for each eligible exchange
- Slippage comparison in basis points
- Arbitrage opportunity alerts with potential profit %

### Historical Backtesting

Simulate order execution during past market conditions using synthetic order book reconstruction.

**Key Capabilities:**

- **Time-Travel Simulation**: Analyze how orders would have performed historically
- **Synthetic Order Books**: Reconstructed from OHLCV volatility using ATR modeling
- **Risk Analysis**: Identifies high-risk periods (>200 bps slippage) and optimal windows (<50 bps)
- **Volatility Profiling**: Tracks spread and volatility percentiles across the backtest period

**Example Query:**

```
"How would buying 500 SOL have performed over the last 7 days?"
```

**Agent Response:**

- Average, max, and min slippage in basis points
- Number of high-risk vs optimal execution periods
- Average fill price and theoretical total cost
- Volatility profile (avg/max spread, volatility percentile)

### Production Deployment

Docker-based deployment with multi-service orchestration for production environments.

**Infrastructure:**

- **Multi-Stage Dockerfile**: Security-hardened with non-root execution
- **4-Service Orchestration**: API, Frontend, Redis, PostgreSQL
- **Health Checks**: Automatic monitoring and restart on failure
- **Graceful Shutdown**: Proper cleanup of exchange connections

**Quick Start:**

```bash
docker-compose up -d
```

See [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) for complete deployment guide.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Required environment variables:

- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `EXCHANGE_API_KEY`: Exchange API key (if using authenticated endpoints)
- `EXCHANGE_API_SECRET`: Exchange API secret

## Usage

### Start the FastAPI backend

```bash
uvicorn market_liquidity_monitor.api.main:app --reload
```

### Run the Streamlit frontend

```bash
streamlit run frontend/app.py
```

### Example Queries

- "What is the SOL liquidity like on Binance?"
- "Show me the order book depth for BTC/USDT"
- "Is there enough liquidity to execute a 10 ETH sell order?"

## Project Structure

```
market_liquidity_monitor/
├── api/                    # FastAPI backend
│   ├── main.py            # API entry point
│   ├── routes.py          # API endpoints
│   └── dependencies.py    # Dependency injection
├── agents/                 # LLM reasoning layer
│   ├── market_agent.py    # Pydantic-AI agent
│   └── tools.py           # Agent tool definitions
├── data_engine/            # Market data layer
│   ├── exchange.py        # CCXT wrapper & Circuit Breaker integration
│   ├── models.py          # Pydantic data models
│   ├── circuit_breaker.py # Fault tolerance state machine
│   ├── historical.py      # Historical data & anomaly detection
│   └── database.py        # Database connection & persistence
├── frontend/               # Streamlit UI
│   ├── app.py             # Legacy chat interface
│   └── enhanced_app.py    # Main dashboard with Alerts & Health
├── config/                 # Configuration
│   └── settings.py        # Environment config
└── tests/                  # Test suite
```

## Security Considerations

- Never commit API keys to version control
- Use environment variables for all secrets
- Validate all user inputs before processing
- Implement rate limiting for production
- Use read-only API keys when possible

## Future Enhancements

- Support for more exchanges
- Advanced liquidity metrics (slippage estimation, market impact) - **Partially Implemented**
- Portfolio optimization based on liquidity
- User authentication and multi-tenancy
