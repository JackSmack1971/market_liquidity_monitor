# Market Liquidity Monitor - Quick Start Guide

Get up and running in 5 minutes with this comprehensive guide.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- OpenRouter API key ([Get one here](https://openrouter.ai/))

## Step 1: Installation

```bash
# Navigate to the project directory
cd market_liquidity_monitor

# Install dependencies
pip install -r requirements.txt

# For testing (optional)
pip install -r tests/requirements-test.txt
```

## Step 2: Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API key
# Required: OPENROUTER_API_KEY
# Optional: EXCHANGE_API_KEY, EXCHANGE_API_SECRET (only if using authenticated endpoints)
```

**Minimum required configuration in `.env`:**

```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

## Step 3: Run the Backend (Option A - FastAPI Server)

If you want to use the REST API:

```bash
# Start the FastAPI server
python -m market_liquidity_monitor.api.main

# Or with uvicorn directly
uvicorn market_liquidity_monitor.api.main:app --reload
```

The API will be available at:

- **Swagger Docs**: <http://localhost:8000/docs>
- **Health Check**: <http://localhost:8000/api/v1/health>

## Step 3: Run the Frontend (Option B - Streamlit App)

For the interactive chat interface:

```bash
# Start the Streamlit app
streamlit run frontend/app.py
```

The app will open in your browser at <http://localhost:8501>

## Step 4: Try It Out

### Using the Streamlit Interface

1. Open <http://localhost:8501>
2. Type a natural language query in the chat:
   - "What is the SOL liquidity like?"
   - "Show me BTC order book depth on Binance"
   - "Can I execute a $10,000 ETH sell order without slippage?"
3. View the AI-powered analysis and real-time order book visualization

### Using the REST API

**Fetch Order Book:**

```bash
curl "http://localhost:8000/api/v1/orderbook/SOL/USDT?exchange=binance&levels=20"
```

**Analyze Liquidity (Natural Language):**

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How is the SOL liquidity on Binance?",
    "symbol": "SOL/USDT",
    "exchange": "binance"
  }'
```

**Quick Liquidity Check:**

```bash
curl "http://localhost:8000/api/v1/quick-check/BTC/USDT?exchange=binance"
```

**Estimate Slippage:**

```bash
curl -X POST "http://localhost:8000/api/v1/estimate-slippage?symbol=ETH/USDT&order_size_usd=5000&side=buy&exchange=binance"
```

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                         User Query                           │
│          "What is the SOL liquidity like?"                   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Pydantic-AI Agent                         │
│  1. Parses query intent                                      │
│  2. Decides to call get_order_book_depth tool               │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   CCXT Data Engine                           │
│  - Fetches real-time order book from Binance                │
│  - Returns structured OrderBook object                       │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM Analysis (via OpenRouter)             │
│  - Analyzes bid-ask spread                                   │
│  - Calculates order book depth                               │
│  - Identifies liquidity risks (thin books, walls)            │
│  - Synthesizes human-readable insights                       │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Structured Response                        │
│  {                                                           │
│    "liquidity_score": "HIGH",                                │
│    "reasoning": "SOL/USDT shows tight spreads...",          │
│    "spread": 0.002,                                          │
│    "bid_depth_10": 45230.5,                                  │
│    ...                                                       │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

## Architecture Components

### 1. Data Engine (CCXT)

- **File**: `data_engine/exchange.py`
- **Purpose**: Fetches real-time order book data from exchanges
- **Key Features**:
  - Async operations (non-blocking)
  - Multi-exchange support (Binance, Coinbase, Kraken, etc.)
  - Precision handling for exchange-specific requirements

### 2. Reasoning Agent (Pydantic-AI)

- **File**: `agents/market_agent.py`
- **Purpose**: LLM-powered analysis of market data
- **Key Features**:
  - Tool calling (can invoke market data functions)
  - Type-safe outputs via Pydantic models
  - Synthesized reasoning beyond raw numbers

### 3. Backend (FastAPI)

- **File**: `api/main.py`
- **Purpose**: REST API server
- **Key Features**:
  - Async endpoints
  - Dependency injection
  - Auto-generated OpenAPI docs

### 4. Frontend (Streamlit)

- **File**: `frontend/app.py`
- **Purpose**: Interactive chat interface
- **Key Features**:
  - Natural language queries
  - Real-time order book visualization
  - Session state management

## Example Use Cases

### 1. Pre-Trade Analysis

**Query**: "I want to buy $50,000 worth of SOL. What's the slippage?"

**What Happens**:

1. Agent fetches SOL/USDT order book
2. Calculates cumulative volume at different price levels
3. Estimates price impact and slippage
4. Provides liquidity score and recommendations

### 2. Market Monitoring

**Query**: "Show me the order book depth for BTC/USDT"

**What Happens**:

1. Fetches real-time order book
2. Displays interactive depth chart
3. Shows key metrics (spread, depth, liquidity zones)

### 3. Exchange Comparison

**Query**: "Which exchange has better ETH liquidity, Binance or Coinbase?"

**What Happens**:

1. Fetches order books from both exchanges
2. Compares spread, depth, and liquidity metrics
3. Provides side-by-side analysis

## Troubleshooting

### Error: "Missing OPENROUTER_API_KEY"

- Make sure you've created a `.env` file (copy from `.env.example`)
- Verify your OpenRouter API key is valid

### Error: "Failed to fetch order book"

- Check your internet connection
- Some exchanges may have rate limits
- Try a different exchange or symbol

### Streamlit app not loading data

- Ensure you're running from the correct directory
- Check that `market_liquidity_monitor` is in your Python path

### Import errors

- Verify all dependencies are installed: `pip install -r requirements.txt`
- Make sure you're using Python 3.9+

## Advanced Configuration

### Using Different LLM Models

Edit `.env`:

```bash
# Use GPT-4o instead of Claude
DEFAULT_MODEL=openai/gpt-4o

# Use Claude Opus for highest quality
DEFAULT_MODEL=anthropic/claude-opus-4

# Use Haiku for faster/cheaper responses
DEFAULT_MODEL=anthropic/claude-3-haiku
```

### Changing Default Exchange

Edit `.env`:

```bash
DEFAULT_EXCHANGE=coinbase
```

### Adding Exchange Authentication

For accessing private endpoints (optional):

```bash
EXCHANGE_API_KEY=your_api_key
EXCHANGE_API_SECRET=your_api_secret
```

## Running Tests

```bash
# Run all tests
pytest market_liquidity_monitor/tests/ -v

# Run with coverage
pytest market_liquidity_monitor/tests/ --cov=market_liquidity_monitor

# Run specific test file
pytest market_liquidity_monitor/tests/test_exchange.py -v
```

## Next Steps

- Explore the API docs at <http://localhost:8000/docs>
- Try different exchanges and trading pairs
- Customize the LLM prompts in `agents/market_agent.py`
- Add new analysis tools in `agents/tools.py`
- Build custom visualizations in the Streamlit app

## Security Best Practices

- Never commit your `.env` file to version control
- Use read-only API keys when possible
- Implement rate limiting for production deployments
- Validate all user inputs before processing

## Need Help?

- Check the main README.md for detailed architecture
- Review the code comments and docstrings
- Open an issue on GitHub
- Consult CCXT docs: <https://docs.ccxt.com>
- Pydantic-AI docs: <https://ai.pydantic.dev>
