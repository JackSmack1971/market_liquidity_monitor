# Market Liquidity Monitor

A system that combines real-time market data with LLM reasoning to monitor liquidity and order book depth.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface (Streamlit)                │
│  - Chat interface for natural language queries               │
│  - Real-time order book visualization                        │
│  - Fragmented updates for performance                        │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                         │
│  - Async API endpoints                                       │
│  - Dependency injection for exchange/agent management        │
│  - Secure secrets management                                 │
└────────────┬────────────────────────────────────────────────┘
             │
             ├───────────┬─────────────┐
             ▼           ▼             ▼
┌──────────────┐  ┌────────────┐  ┌─────────────┐
│ Data Engine  │  │ LLM Agent  │  │   Config    │
│   (CCXT)     │  │(Pydantic-AI)│  │ Management  │
└──────────────┘  └────────────┘  └─────────────┘
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
